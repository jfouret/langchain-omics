"""Unified SRA client using EBI Search and ENA Portal APIs."""

import logging
import time
from typing import Any, Dict, List, Optional, ClassVar
from urllib.parse import urlencode

import requests
from pydantic import BaseModel, Field, ConfigDict


class UnifiedSRAClient(BaseModel):
    """Unified SRA client using EBI Search for data + ENA Portal for field discovery.
    
    This client combines the strengths of both APIs:
    - EBI Search: Simple queries, efficient cross-references, entry retrieval
    - ENA Portal: Field discovery to know what fields are available
    """
    
    ebi_search_base_url: str = "https://www.ebi.ac.uk/ebisearch/ws/rest"
    ena_portal_base_url: str = "https://www.ebi.ac.uk/ena/portal/api"
    
    headers: Dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "langchain-omics/1.0.0",
            "Accept": "application/json",
        }
    )
    request_timeout: int = 30
    rate_limit_delay: float = 0.1
    log_level: str = "WARNING"  # DEBUG, INFO, WARNING, ERROR
    
    # Cache for field information
    field_cache: Dict[str, List[str]] = Field(default_factory=dict)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Setup logging (use object.__setattr__ to bypass Pydantic validation)
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.log_level.upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        object.__setattr__(self, 'logger', logger)
    
    # Domain mappings for field discovery
    FIELD_DISCOVERY_MAPPING: ClassVar[Dict[str, str]] = {
        "sra-study": "study",
        "sra-experiment": "read_experiment", 
        "sra-sample": "sample",
        "sra-run": "read_run"
    }
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Make HTTP request with error handling and rate limiting."""
        if params:
            clean_params = {k: str(v) for k, v in params.items() if v is not None}
            if clean_params:
                url = f"{url}?{urlencode(clean_params)}"
        
        time.sleep(self.rate_limit_delay)
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.request_timeout)
            
            # Handle 204 No Content (end of pagination)
            if response.status_code == 204:
                return []
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"API request failed: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid JSON response: {e}")

    def discover_fields(self, sra_domain: str) -> List[str]:
        """Use ENA Portal to discover available fields for SRA domain (cached).
        
        Args:
            sra_domain: SRA domain (e.g., 'sra-study', 'sra-sample')
            
        Returns:
            List of available field names
        """
        if sra_domain in self.field_cache:
            return self.field_cache[sra_domain]
        
        # Map SRA domain to ENA Portal result type
        ena_result_type = self.FIELD_DISCOVERY_MAPPING.get(sra_domain)
        if not ena_result_type:
            return []
        
        # Get fields from ENA Portal
        url = f"{self.ena_portal_base_url}/returnFields"
        params = {"result": ena_result_type, "format": "json"}
        
        try:
            response = self._make_request(url, params)
            if isinstance(response, list):
                fields = [field.get("columnId", "") for field in response if field.get("columnId")]
                self.field_cache[sra_domain] = fields
                return fields
        except Exception:
            # If field discovery fails, return empty list
            pass
        
        return []

    def search_domain(self, domain: str, query: str, size: int = 1000) -> List[Dict[str, Any]]:
        """Use EBI Search for simple domain search.
        
        Args:
            domain: SRA domain (e.g., 'sra-study', 'sra-sample')
            query: Search query string
            size: Maximum number of results
            
        Returns:
            List of entry dictionaries
        """
        url = f"{self.ebi_search_base_url}/{domain}"
        params = {
            "query": query,
            "size": str(size),
            "format": "json"
        }
        
        try:
            response = self._make_request(url, params)
            return response.get("entries", [])
        except Exception:
            return []

    def get_entries_batch(self, domain: str, entry_ids: List[str], 
                         fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Use EBI Search entry retrieval with discovered fields.
        
        Args:
            domain: SRA domain
            entry_ids: List of entry IDs (max 100 per batch)
            fields: List of fields to retrieve (if None, use discovered fields)
            
        Returns:
            List of entry dictionaries with full field data
        """
        if not entry_ids:
            return []
        
        # Limit to 100 entries per batch (EBI Search limit)
        batch_ids = entry_ids[:100]
        
        # Use discovered fields if none specified
        if fields is None:
            fields = self.discover_fields(domain)
        
        # Build URL and parameters
        url = f"{self.ebi_search_base_url}/{domain}/entry/{','.join(batch_ids)}"
        params = {"format": "json"}
        
        if fields:
            params["fields"] = ",".join(fields)
        
        try:
            response = self._make_request(url, params)
            entries = response.get("entries", [])
            
            # Convert field format and clean empty fields
            cleaned_entries = []
            for entry in entries:
                cleaned_entry = self._process_entry_fields(entry)
                if cleaned_entry:  # Only add non-empty entries
                    cleaned_entries.append(cleaned_entry)
            
            return cleaned_entries
        except Exception:
            return []

    def get_entries_paginated(self, domain: str, entry_ids: List[str], 
                            fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get entries in batches of 100 (EBI Search limit).
        
        Args:
            domain: SRA domain
            entry_ids: List of entry IDs (any size)
            fields: List of fields to retrieve
            
        Returns:
            List of all entry dictionaries
        """
        all_entries = []
        
        # Process in batches of 100
        for i in range(0, len(entry_ids), 100):
            batch_ids = entry_ids[i:i+100]
            batch_entries = self.get_entries_batch(domain, batch_ids, fields)
            all_entries.extend(batch_entries)
        
        return all_entries

    def get_cross_references(self, domain: str, entry_id: str, target_domain: str) -> List[str]:
        """Use EBI Search for cross-references (single entry).
        
        Args:
            domain: Source domain (e.g., 'sra-sample')
            entry_id: Source entry ID
            target_domain: Target domain (e.g., 'sra-experiment')
            
        Returns:
            List of cross-referenced entry IDs
        """
        return self.get_cross_references_batch(domain, [entry_id], target_domain).get(entry_id, [])

    def get_cross_references_batch(self, domain: str, entry_ids: List[str], target_domain: str, chunk_size: int = 25) -> Dict[str, List[str]]:
        """Use EBI Search for cross-references (batch of entries with chunking and pagination).
        
        Args:
            domain: Source domain (e.g., 'sra-sample')
            entry_ids: List of source entry IDs (any size)
            target_domain: Target domain (e.g., 'sra-experiment')
            chunk_size: Size of chunks for API calls (default 100, max 100)
            
        Returns:
            Dictionary mapping entry_id -> list of cross-referenced IDs
        """
        if not entry_ids:
            return {}
        
        # Ensure chunk_size doesn't exceed API limit
        chunk_size = min(chunk_size, 100)
        
        all_results = {}
        entries_needing_pagination = []
        
        # Process in chunks
        for i in range(0, len(entry_ids), chunk_size):
            batch_ids = entry_ids[i:i+chunk_size]
            
            url = f"{self.ebi_search_base_url}/{domain}/entry/{','.join(batch_ids)}/xref/{target_domain}"
            params = {"format": "json", "size": "100"}  # Get first 100 references
            
            self.logger.debug(f"Cross-reference batch call: {domain} -> {target_domain}, {len(batch_ids)} entries (chunk {i//chunk_size + 1})")
            
            try:
                response = self._make_request(url, params)
                entries = response.get("entries", [])
                
                # Build mapping of source_id -> [target_ids] and check for pagination needs
                for entry in entries:
                    source_id = entry.get("source")
                    if source_id:
                        cross_refs = []
                        references = entry.get("references", [])
                        for ref in references:
                            acc = ref.get("acc")
                            if acc:
                                cross_refs.append(acc)
                        
                        all_results[source_id] = cross_refs
                        
                        # Check if this entry has more references than we got
                        reference_count = entry.get("referenceCount", 0)
                        if reference_count > 100:
                            self.logger.debug(f"Entry {source_id} has {reference_count} references, needs pagination")
                            entries_needing_pagination.append((source_id, reference_count))
                
                # Ensure all requested IDs are in result (even if empty)
                for entry_id in batch_ids:
                    if entry_id not in all_results:
                        all_results[entry_id] = []
                        
            except Exception as e:
                self.logger.debug(f"Cross-reference batch chunk failed: {e}")
                # Add empty results for failed chunk
                for entry_id in batch_ids:
                    if entry_id not in all_results:
                        all_results[entry_id] = []
        
        # Handle entries that need pagination (have more than 100 references)
        for source_id, reference_count in entries_needing_pagination:
            self.logger.debug(f"Paginating references for {source_id} ({reference_count} total)")
            
            # Get remaining references in batches of 100
            start = 100  # We already got 0-99
            while start < reference_count:
                url = f"{self.ebi_search_base_url}/{domain}/entry/{source_id}/xref/{target_domain}"
                params = {"format": "json", "size": "100", "start": str(start)}
                
                try:
                    response = self._make_request(url, params)
                    entries = response.get("entries", [])
                    
                    for entry in entries:
                        if entry.get("source") == source_id:
                            references = entry.get("references", [])
                            for ref in references:
                                acc = ref.get("acc")
                                if acc:
                                    all_results[source_id].append(acc)
                    
                    start += 100
                    
                except Exception as e:
                    self.logger.debug(f"Pagination failed for {source_id} at start={start}: {e}")
                    break  # Stop pagination for this entry if it fails
        
        return all_results

    def search_multi_level(self, query: str, max_studies: Optional[int] = None) -> List[str]:
        """Perform multi-level search to find all relevant study accessions.
        
        This implements the workflow from brief.md:
        1. Search study, experiment, and sample domains
        2. Use cross-references to traverse to parent studies
        3. Return unique list of study accessions
        
        Args:
            query: Search query string
            max_studies: Maximum number of studies to return
            
        Returns:
            List of unique study accessions
        """
        self.logger.debug(f"Starting multi-level search for query: '{query}'")
        study_accessions = set()
        
        # Step 1: Direct study search
        self.logger.debug("Step 1: Direct study search")
        study_results = self.search_domain("sra-study", query, size=100)  # Limit for performance
        for study in study_results:
            acc = study.get("acc") or study.get("id")
            if acc:
                study_accessions.add(acc)
        self.logger.debug(f"Found {len(study_accessions)} studies from direct search")
        
        # Early exit if we have enough studies
        if max_studies and len(study_accessions) >= max_studies:
            return list(study_accessions)[:max_studies]
        
        # Step 2: Search experiments and get their studies via cross-references (batched)
        self.logger.debug("Step 2: Experiment -> Study cross-references")
        exp_results = self.search_domain("sra-experiment", query, size=50)  # Limit for performance
        exp_ids = [exp.get("acc") or exp.get("id") for exp in exp_results if exp.get("acc") or exp.get("id")]
        
        if exp_ids:
            # Batch cross-reference call for experiments -> studies
            exp_to_studies = self.get_cross_references_batch("sra-experiment", exp_ids, "sra-study")
            for exp_id, study_refs in exp_to_studies.items():
                study_accessions.update(study_refs)
        
        self.logger.debug(f"Found {len(study_accessions)} studies after experiment search")
        
        # Early exit if we have enough studies
        if max_studies and len(study_accessions) >= max_studies:
            return list(study_accessions)[:max_studies]
        
        # Step 3: Search samples and get their studies via experiments (batched)
        self.logger.debug("Step 3: Sample -> Experiment -> Study cross-references")
        sample_results = self.search_domain("sra-sample", query, size=30)  # Limit for performance
        sample_ids = [sample.get("acc") or sample.get("id") for sample in sample_results if sample.get("acc") or sample.get("id")]
        
        if sample_ids:
            # Batch cross-reference call for samples -> experiments
            sample_to_exps = self.get_cross_references_batch("sra-sample", sample_ids, "sra-experiment")
            
            # Collect all experiment IDs from samples
            all_exp_ids_from_samples = set()
            for sample_id, exp_refs in sample_to_exps.items():
                all_exp_ids_from_samples.update(exp_refs)
            
            # Batch cross-reference call for experiments -> studies
            if all_exp_ids_from_samples:
                exp_to_studies = self.get_cross_references_batch("sra-experiment", list(all_exp_ids_from_samples), "sra-study")
                for exp_id, study_refs in exp_to_studies.items():
                    study_accessions.update(study_refs)
        
        self.logger.debug(f"Found {len(study_accessions)} total unique studies")
        
        # Convert to list and limit if requested
        study_list = list(study_accessions)
        if max_studies is not None:
            study_list = study_list[:max_studies]
        
        return study_list

    def get_full_study_data(self, study_accessions: List[str]) -> List[Dict[str, Any]]:
        """Retrieve full nested data for studies.
        
        Args:
            study_accessions: List of study accession IDs
            
        Returns:
            List of complete study objects with nested structure
        """
        if not study_accessions:
            return []
        
        # Get full data for studies
        studies = self.get_entries_paginated("sra-study", study_accessions)
        
        # Collect all related entity IDs
        all_sample_ids = set()
        all_experiment_ids = set()
        all_run_ids = set()
        
        # For each study, find all related samples, experiments, and runs
        for study in studies:
            study_acc = study.get("acc") or study.get("id")
            if study_acc:
                # Get samples for this study (via experiments)
                exp_ids = self.get_cross_references("sra-study", study_acc, "sra-experiment")
                all_experiment_ids.update(exp_ids)
                
                for exp_id in exp_ids:
                    # Get samples for each experiment
                    sample_ids = self.get_cross_references("sra-experiment", exp_id, "sra-sample")
                    all_sample_ids.update(sample_ids)
                    
                    # Get runs for each experiment
                    run_ids = self.get_cross_references("sra-experiment", exp_id, "sra-run")
                    all_run_ids.update(run_ids)
        
        # Get full data for all related entities
        samples = self.get_entries_paginated("sra-sample", list(all_sample_ids))
        experiments = self.get_entries_paginated("sra-experiment", list(all_experiment_ids))
        runs = self.get_entries_paginated("sra-run", list(all_run_ids))
        
        # Build nested structure
        return self._assemble_nested_structure(studies, samples, experiments, runs)

    def _process_entry_fields(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Process entry fields from EBI Search format and clean empty values.
        
        Args:
            entry: Raw entry from EBI Search
            
        Returns:
            Cleaned entry dictionary
        """
        processed = {}
        
        # Keep basic entry info
        for key in ["acc", "id", "source"]:
            value = entry.get(key)
            if value:
                processed[key] = value
        
        # Process fields array into flat dictionary
        fields = entry.get("fields", [])
        if isinstance(fields, list):
            for field in fields:
                field_id = field.get("id")
                field_values = field.get("values", [])
                
                if field_id and field_values:
                    # Use single value if only one, otherwise keep as list
                    if len(field_values) == 1:
                        processed[field_id] = field_values[0]
                    else:
                        processed[field_id] = field_values
        
        # Clean empty, null, or whitespace-only fields
        return self._clean_fields(processed)

    def _clean_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove empty, null, or whitespace-only fields from data.
        
        Args:
            data: Dictionary to clean
            
        Returns:
            Cleaned dictionary
        """
        cleaned = {}
        for key, value in data.items():
            if value is not None and value != "" and str(value).strip():
                # For lists, filter out empty values
                if isinstance(value, list):
                    clean_list = [v for v in value if v is not None and str(v).strip()]
                    if clean_list:
                        cleaned[key] = clean_list
                else:
                    cleaned[key] = value
        return cleaned

    def _assemble_nested_structure(self, studies: List[Dict], samples: List[Dict], 
                                  experiments: List[Dict], runs: List[Dict]) -> List[Dict[str, Any]]:
        """Assemble flat data into nested Study -> Sample -> Experiment -> Run structure.
        
        Args:
            studies: List of study records
            samples: List of sample records
            experiments: List of experiment records
            runs: List of run records
            
        Returns:
            List of nested study objects
        """
        # Index data by accessions for efficient lookup
        study_map = {}
        sample_map = {}
        experiment_map = {}
        run_map = {}
        
        # Build lookup maps
        for study in studies:
            acc = study.get("acc") or study.get("id")
            if acc:
                study_map[acc] = study
        
        for sample in samples:
            acc = sample.get("acc") or sample.get("id")
            if acc:
                sample_map[acc] = sample
        
        for experiment in experiments:
            acc = experiment.get("acc") or experiment.get("id")
            if acc:
                experiment_map[acc] = experiment
        
        for run in runs:
            acc = run.get("acc") or run.get("id")
            if acc:
                run_map[acc] = run
        
        # Build nested structure using cross-references
        result_studies = []
        
        for study_acc, study_data in study_map.items():
            nested_study = dict(study_data)
            nested_study["samples"] = []
            
            # Get experiments for this study
            study_experiments = self.get_cross_references("sra-study", study_acc, "sra-experiment")
            
            # Group experiments by sample
            sample_to_experiments = {}
            for exp_acc in study_experiments:
                if exp_acc in experiment_map:
                    # Get samples for this experiment
                    exp_samples = self.get_cross_references("sra-experiment", exp_acc, "sra-sample")
                    for sample_acc in exp_samples:
                        if sample_acc not in sample_to_experiments:
                            sample_to_experiments[sample_acc] = []
                        sample_to_experiments[sample_acc].append(exp_acc)
            
            # Build sample -> experiment -> run structure
            for sample_acc, exp_accs in sample_to_experiments.items():
                if sample_acc in sample_map:
                    nested_sample = dict(sample_map[sample_acc])
                    nested_sample["experiments"] = []
                    
                    for exp_acc in exp_accs:
                        if exp_acc in experiment_map:
                            nested_experiment = dict(experiment_map[exp_acc])
                            nested_experiment["runs"] = []
                            
                            # Get runs for this experiment
                            exp_runs = self.get_cross_references("sra-experiment", exp_acc, "sra-run")
                            for run_acc in exp_runs:
                                if run_acc in run_map:
                                    nested_experiment["runs"].append(run_map[run_acc])
                            
                            nested_sample["experiments"].append(nested_experiment)
                    
                    nested_study["samples"].append(nested_sample)
            
            result_studies.append(nested_study)
        
        return result_studies

    def search_and_retrieve(self, query: str, max_studies: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run the complete workflow: search -> cross-reference -> retrieve -> assemble.
        
        This is the main method that combines all steps:
        1. Multi-level search to find relevant studies
        2. Get full nested data for those studies
        3. Return complete study objects with samples, experiments, and runs
        
        Args:
            query: Search query string (e.g., "respiratory syncytial virus", "RSV")
            max_studies: Maximum number of studies to return
            
        Returns:
            List of complete study objects with nested structure
            
        Example:
            >>> client = UnifiedSRAClient()
            >>> studies = client.search_and_retrieve("RSV", max_studies=5)
            >>> for study in studies:
            ...     print(f"Study: {study.get('acc')} - {study.get('title', '')[:50]}...")
            ...     print(f"  Samples: {len(study.get('samples', []))}")
        """
        # Step 1: Find all relevant study accessions
        study_accessions = self.search_multi_level(query, max_studies)
        
        if not study_accessions:
            return []
        
        # Step 2: Get full nested data for those studies
        return self.get_full_study_data(study_accessions)
