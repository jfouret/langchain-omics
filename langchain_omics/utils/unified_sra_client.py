"""Unified SRA client using EBI Search and ENA Portal APIs."""

import logging
import time
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode

import json

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field


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
    rate_limit_delay: float = 0
    log_level: str = "WARNING"  # DEBUG, INFO, WARNING, ERROR

    # Cache for field information
    field_cache: Dict[str, List[Dict[str, str]]] = dict()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Setup logging (use object.__setattr__ to bypass Pydantic validation)
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.log_level.upper()))
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        object.__setattr__(self, "logger", logger)

    # Domain mappings for field discovery
    FIELD_DISCOVERY_MAPPING: ClassVar[Dict[str, str]] = {
        "sra-study": "study",
        "sra-experiment": "read_experiment",
        "sra-sample": "sample",
        "sra-run": "read_run",
    }

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Make HTTP request with error handling and rate limiting."""
        if params:
            clean_params = {k: str(v) for k, v in params.items() if v is not None}
            if clean_params:
                url = f"{url}?{urlencode(clean_params)}"

        self.logger.log(5, f"TRACE: Making request to {url}")

        time.sleep(self.rate_limit_delay)

        try:
            response = requests.get(
                url, headers=self.headers, timeout=self.request_timeout
            )

            # Handle 204 No Content (end of pagination)
            if response.status_code == 204:
                return []

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"API request failed: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid JSON response: {e}")

    def discover_fields(self, sra_domain: str) -> Dict[str, str]:
        """Use EBI Search to discover available retrievable fields for SRA domain
        (cached).

        Args:
            sra_domain: SRA domain (e.g., 'sra-study', 'sra-sample')

        Returns:
            List of retrievable field names from EBI Search
        """
        if sra_domain in self.field_cache.keys():
            return self.field_cache[sra_domain]
        
        sra_type = self.FIELD_DISCOVERY_MAPPING[sra_domain]

        self.logger.debug(f"Discovering fields for domain: {sra_domain}")

        # Get domain metadata from EBI Search
        url = f"{self.ena_portal_base_url}/returnFields"

        params = {
            "format": "json",
            "dataPortal": "ena",
            "result": sra_type    
        }

        response = self._make_request(url, params)
        fields = dict()
        for field in response:
            fields[field["columnId"]] = field["description"]
        self.field_cache[sra_domain] = fields
        return fields

    def search_domain(
        self, domain: str, query: str, size: int = 100
    ) -> List[Dict[str, Any]]:
        """Use EBI Search for simple domain search.

        Args:
            domain: SRA domain (e.g., 'sra-study', 'sra-sample')
            query: Search query string
            size: Maximum number of results

        Returns:
            List of entry dictionaries
        """
        self.logger.debug(
            f"Searching domain {domain} with query '{query}', size={size}"
        )

        url = f"{self.ebi_search_base_url}/{domain}"
        params = {"query": query, "size": str(size), "format": "json"}

        try:
            response = self._make_request(url, params)
            entries = response.get("entries", [])
            self.logger.debug(
                f"Domain search returned {len(entries)} entries for {domain}"
            )
            return entries
        except Exception as e:
            self.logger.debug(f"Domain search failed for {domain}: {e}")
            return []

    def get_entries_batch(
        self, domain: str, entry_ids: List[str], fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
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
        assert len(entry_ids) <= 100

        self.logger.debug(
            f"Getting entries batch for {domain}: {len(entry_ids)} entries"
        )

        # Use discovered fields if none specified
        if fields is None:
            fields = self.discover_fields(domain)
            self.logger.debug(f"Using {len(fields)} discovered fields for {domain}")

        # Build URL and parameters
        url = f"{self.ebi_search_base_url}/{domain}/entry/{','.join(entry_ids)}"
        params = {"format": "json"}

        if fields:
            params["fields"] = ",".join(fields.keys())

        try:
            response = self._make_request(url, params)
            entries = response.get("entries", [])
            self.logger.debug(f"Retrieved {len(entries)} raw entries for {domain}")

            # Convert field format and clean empty fields
            cleaned_entries = []
            for entry in entries:
                self.logger.debug(
                    f"Processing entry: {entry.get('id', 'unknown')}"
                    f"{json.dumps(entry, indent=2)}"
                )                
                cleaned_entry = self._process_entry_fields(entry)
                if cleaned_entry:  # Only add non-empty entries
                    cleaned_entries.append(cleaned_entry)

            self.logger.debug(
                f"Processed {len(cleaned_entries)} cleaned entries for {domain}"
            )
            return cleaned_entries
        except Exception as e:
            self.logger.error(f"Get entries batch failed for {domain}: {e}")
            exit(1)

    def get_entries_paginated(
        self, domain: str, entry_ids: List[str], fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get entries in batches of 100 (EBI Search limit).

        Args:
            domain: SRA domain
            entry_ids: List of entry IDs (any size)
            fields: List of fields to retrieve

        Returns:
            List of all entry dictionaries
        """
        if not entry_ids:
            return []

        self.logger.debug(
            f"Getting entries paginated for {domain}: {len(entry_ids)} total entries"
        )
        all_entries = []

        # Process in batches of 100
        for i in range(0, len(entry_ids), 100):
            batch_ids = entry_ids[i : i + 100]
            self.logger.debug(
                f"Processing batch from {i} for {domain}: {len(batch_ids)} entries"
            )
            batch_entries = self.get_entries_batch(domain, batch_ids, fields)
            all_entries.extend(batch_entries)

        self.logger.debug(f"Retrieved {len(all_entries)} total entries for {domain}")
        return all_entries

    def get_cross_references(
        self, domain: str, entry_id: str, target_domain: str
    ) -> List[str]:
        """Use EBI Search for cross-references (single entry).

        Args:
            domain: Source domain (e.g., 'sra-sample')
            entry_id: Source entry ID
            target_domain: Target domain (e.g., 'sra-experiment')

        Returns:
            List of cross-referenced entry IDs
        """
        return self.get_cross_references_batch(domain, [entry_id], target_domain).get(
            entry_id, []
        )

    def get_cross_references_batch(
        self,
        domain: str,
        entry_ids: List[str],
        target_domain: str,
        chunk_size: int = 100,
    ) -> Dict[str, List[str]]:
        """Use EBI Search for cross-references (batch of entries with chunking 
        and pagination).

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
            batch_ids = entry_ids[i : i + chunk_size]

            url = (
                f"{self.ebi_search_base_url}/{domain}/entry/"
                f"{','.join(batch_ids)}/xref/{target_domain}"
            )
            params = {"format": "json", "size": "100"}  # Get first 100 references

            self.logger.debug(
                f"Cross-reference batch call: {domain} -> {target_domain}, "
                f"{len(batch_ids)} entries (chunk {i//chunk_size + 1})"
            )

            try:
                response = self._make_request(url, params)
                entries = response.get("entries", [])

                # Build mapping of source_id -> [target_ids] and check for pagination
                for entry in entries:
                    # The source_id is actually the acc/id of the entry we queried
                    source_id = entry.get("acc") or entry.get("id")
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
                            self.logger.debug(
                                f"Entry {source_id} has {reference_count} "
                                "references, needs pagination"
                            )
                            entries_needing_pagination.append(
                                (source_id, reference_count)
                            )

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
            self.logger.debug(
                f"Paginating references for {source_id} ({reference_count} total)"
            )

            # Get remaining references in batches of 100
            start = 100  # We already got 0-99
            while start < reference_count:
                url = (
                    f"{self.ebi_search_base_url}/{domain}/entry/{source_id}/"
                    f"xref/{target_domain}"
                )
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
                    self.logger.debug(
                        f"Pagination failed for {source_id} at start={start}: {e}"
                    )
                    break  # Stop pagination for this entry if it fails

        return all_results

    def search_multiple_domains(
        self,
        databases: List[str],
        query: str,
        size: int = 100,
        size_per_db: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple databases and return results by database.

        Args:
            databases: List of database names to search
                (e.g., ["sra-study", "sra-sample"])
            query: Search query string
            size: Default maximum number of results per database
                (used if not in size_per_db)
            size_per_db: Optional dictionary mapping database name to custom size limit
                        If database is not in keys, uses the default size parameter

        Returns:
            Dictionary mapping database name to list of entry dictionaries.
            Maintains search order by processing databases in the order provided.
        """
        self.logger.debug(f"Searching multiple databases: {databases}")
        results = {}

        for database in databases:
            # Determine size for this database
            db_size = size_per_db.get(database, size) if size_per_db else size

            self.logger.debug(f"Searching database: {database} with size={db_size}")
            entries = self.search_domain(database, query, size=db_size)
            results[database] = entries
            self.logger.debug(f"Found {len(entries)} entries in {database}")

        return results

    def search_multi_level(
        self, query: str, max_studies: Optional[int] = None
    ) -> List[str]:
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
        study_results = self.search_domain(
            "sra-study", query, size=100
        )  # Limit for performance
        for study in study_results:
            acc = study.get("acc") or study.get("id")
            if acc:
                study_accessions.add(acc)
        self.logger.debug(f"Found {len(study_accessions)} studies from direct search")

        # Early exit if we have enough studies
        if max_studies and len(study_accessions) >= max_studies:
            return list(study_accessions)[:max_studies]

        # Step 2: Search experiments and get their studies via cross-references(batched)
        self.logger.debug("Step 2: Experiment -> Study cross-references")
        exp_results = self.search_domain(
            "sra-experiment", query, size=50
        )  # Limit for performance
        exp_ids = [
            exp.get("acc") or exp.get("id")
            for exp in exp_results
            if exp.get("acc") or exp.get("id")
        ]

        if exp_ids:
            # Batch cross-reference call for experiments -> studies
            exp_to_studies = self.get_cross_references_batch(
                "sra-experiment", exp_ids, "sra-study"
            )
            for exp_id, study_refs in exp_to_studies.items():
                study_accessions.update(study_refs)

        self.logger.debug(
            f"Found {len(study_accessions)} studies after experiment search"
        )

        # Early exit if we have enough studies
        if max_studies and len(study_accessions) >= max_studies:
            return list(study_accessions)[:max_studies]

        # Step 3: Search samples and get their studies via experiments (batched)
        self.logger.debug("Step 3: Sample -> Experiment -> Study cross-references")
        sample_results = self.search_domain(
            "sra-sample", query, size=30
        )  # Limit for performance
        sample_ids = [
            sample.get("acc") or sample.get("id")
            for sample in sample_results
            if sample.get("acc") or sample.get("id")
        ]

        if sample_ids:
            # Batch cross-reference call for samples -> experiments
            sample_to_exps = self.get_cross_references_batch(
                "sra-sample", sample_ids, "sra-experiment"
            )

            # Collect all experiment IDs from samples
            all_exp_ids_from_samples = set()
            for sample_id, exp_refs in sample_to_exps.items():
                all_exp_ids_from_samples.update(exp_refs)

            # Batch cross-reference call for experiments -> studies
            if all_exp_ids_from_samples:
                exp_to_studies = self.get_cross_references_batch(
                    "sra-experiment", list(all_exp_ids_from_samples), "sra-study"
                )
                for exp_id, study_refs in exp_to_studies.items():
                    study_accessions.update(study_refs)

        self.logger.debug(f"Found {len(study_accessions)} total unique studies")

        # Convert to list and limit if requested
        study_list = list(study_accessions)
        if max_studies is not None:
            study_list = study_list[:max_studies]

        return study_list

    def _get_entity_type_from_db(self, database: str) -> str:
        """Map database name to entity type."""
        return database.replace("sra-", "")

    def _get_db_from_entity_type(self, entity_type: str) -> str:
        """Map entity type to database name."""
        return f"sra-{entity_type}"

    def _get_relationships(self, entity_type: str) -> Dict[str, List[str]]:
        """Get cross-referenced entity types for a given entity type.

        SRA Cross-References (undirected):
        - Study cross-references: Experiments
        - Experiment cross-references: Studies, Samples, Runs
        - Sample cross-references: Experiments
        - Run cross-references: Experiments
        """
        relationships = {
            "study": {"crossref": ["experiment"]},
            "experiment": {"crossref": ["study", "sample", "run"]},
            "sample": {"crossref": ["experiment"]},
            "run": {"crossref": ["experiment"]},
        }
        return relationships.get(entity_type, {"crossref": []})

    def get_full_study_data(self, study_accessions: List[str]) -> List[Dict[str, Any]]:
        """Retrieve full nested data for studies using batch cross-reference calls.

        Args:
            study_accessions: List of study accession IDs

        Returns:
            List of complete study objects with nested structure
        """
        if not study_accessions:
            return []

        self.logger.debug(
            f"Getting full study data for {len(study_accessions)} studies"
        )

        # Get full data for studies
        self.logger.debug("Step 1: Retrieving study entries")
        studies = self.get_entries_paginated("sra-study", study_accessions)

        # Step 1: Batch studies → experiments
        self.logger.debug("Step 2: Getting study → experiment cross-references")
        study_ids = [
            s.get("acc") or s.get("id") for s in studies if s.get("acc") or s.get("id")
        ]
        study_to_experiments = self.get_cross_references_batch(
            "sra-study", study_ids, "sra-experiment"
        )

        # Collect all experiment IDs
        all_experiment_ids = set()
        for exp_list in study_to_experiments.values():
            all_experiment_ids.update(exp_list)
        self.logger.debug(f"Found {len(all_experiment_ids)} unique experiments")

        # Step 2: Batch experiments → samples and runs
        self.logger.debug(
            "Step 3: Getting experiment → sample and experiment → run cross-references"
        )
        exp_ids_list = list(all_experiment_ids)
        experiment_to_samples = self.get_cross_references_batch(
            "sra-experiment", exp_ids_list, "sra-sample"
        )
        experiment_to_runs = self.get_cross_references_batch(
            "sra-experiment", exp_ids_list, "sra-run"
        )

        # Collect all entity IDs
        all_sample_ids = set()
        all_run_ids = set()
        for sample_list in experiment_to_samples.values():
            all_sample_ids.update(sample_list)
        for run_list in experiment_to_runs.values():
            all_run_ids.update(run_list)

        self.logger.debug(
            f"Found {len(all_sample_ids)} unique samples and "
            f"{len(all_run_ids)} unique runs"
        )

        # Get full data for all related entities
        self.logger.debug("Step 4: Retrieving full data for all entities")
        samples = self.get_entries_paginated("sra-sample", list(all_sample_ids))
        experiments = self.get_entries_paginated(
            "sra-experiment", list(all_experiment_ids)
        )
        runs = self.get_entries_paginated("sra-run", list(all_run_ids))

        # Build nested structure with mappings to avoid re-calling APIs
        self.logger.debug("Step 5: Assembling nested structure")
        return self._assemble_nested_structure(
            studies,
            samples,
            experiments,
            runs,
            study_to_experiments,
            experiment_to_samples,
            experiment_to_runs,
        )

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

        # Process fields - EBI Search returns fields as a dictionary
        fields = entry.get("fields", {})
        if isinstance(fields, dict):
            for field_id, field_values in fields.items():
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

    def _assemble_nested_structure(
        self,
        studies: List[Dict],
        samples: List[Dict],
        experiments: List[Dict],
        runs: List[Dict],
        study_to_experiments: Dict[str, List[str]],
        experiment_to_samples: Dict[str, List[str]],
        experiment_to_runs: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Assemble flat data into nested Study -> Experiment -> Sample + 
        Runs structure using provided mappings.

        Args:
            studies: List of study records
            samples: List of sample records
            experiments: List of experiment records
            runs: List of run records
            study_to_experiments: Mapping of study_id -> [experiment_ids]
            experiment_to_samples: Mapping of experiment_id -> [sample_ids]
            experiment_to_runs: Mapping of experiment_id -> [run_ids]

        Returns:
            List of nested study objects with correct SRA hierarchy
        """
        self.logger.debug(
            f"Assembling nested structure: {len(studies)} studies, {len(experiments)} "
            f"experiments, {len(samples)} samples, {len(runs)} runs"
        )

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

        self.logger.debug("Built lookup maps")

        # Build nested structure: Study -> Experiments -> (Sample + Runs)
        result_studies = []

        for study_acc, study_data in study_map.items():
            nested_study = dict(study_data)
            nested_study["experiments"] = []

            # Get experiments for this study using provided mapping
            study_experiments = study_to_experiments.get(study_acc, [])
            self.logger.debug(
                f"Study {study_acc} has {len(study_experiments)} experiments"
            )

            for exp_acc in study_experiments:
                if exp_acc in experiment_map:
                    nested_experiment = dict(experiment_map[exp_acc])

                    # Get the sample for this experiment using provided mapping
                    exp_samples = experiment_to_samples.get(exp_acc, [])
                    if exp_samples and exp_samples[0] in sample_map:
                        nested_experiment["sample"] = sample_map[exp_samples[0]]
                    else:
                        nested_experiment["sample"] = None

                    # Get runs for this experiment using provided mapping
                    nested_experiment["runs"] = []
                    exp_runs = experiment_to_runs.get(exp_acc, [])
                    for run_acc in exp_runs:
                        if run_acc in run_map:
                            nested_experiment["runs"].append(run_map[run_acc])

                    self.logger.debug(
                        f"Experiment {exp_acc} has {len(exp_runs)} runs and sample:"
                        f" {exp_samples[0] if exp_samples else 'None'}"
                    )
                    nested_study["experiments"].append(nested_experiment)

            result_studies.append(nested_study)

        self.logger.debug(f"Assembled {len(result_studies)} nested study objects")
        return result_studies

    def get_cross_ref_recursively(
        self,
        entities: List[Tuple[str, str]],
        target_dbs: Dict[str, List[str]],
        max_depth: int = 2,
    ) -> pd.DataFrame:
        """Recursively build cross-reference relations with depth limit.

        Args:
            entities: List of (ID, DATABASE) tuples to start from
            target_dbs: Dict mapping source_db → list of target databases to query
            max_depth: Maximum recursion depth (default: 2)

        Returns:
            DataFrame with columns [source_id, target_id, source_db, target_db]
        """
        # Initialize shared state for recursion
        relations_df = pd.DataFrame(
            columns=["source_id", "target_id", "source_db", "target_db"]
        )
        visited: Dict[str, List[str]] = {db: [] for db in target_dbs.keys()}

        def recursive_query(
            current_entities: List[Tuple[str, str]], current_depth: int
        ) -> None:
            """Inner function to share state across recursive calls."""
            if current_depth >= max_depth:
                return

            # Group entities by database for batch querying
            entities_by_db: Dict[str, List[str]] = {}
            for entity_id, entity_db in current_entities:
                if entity_db not in entities_by_db:
                    entities_by_db[entity_db] = []
                entities_by_db[entity_db].append(entity_id)

            # Query each database
            for source_db, source_ids in entities_by_db.items():
                if not source_ids:
                    continue

                # Get target databases for this source database
                target_database_list = target_dbs.get(source_db, [])

                for target_db in target_database_list:
                    # Get cross-references in batch
                    cross_refs = self.get_cross_references_batch(
                        source_db, source_ids, target_db
                    )

                    # Add to relations DataFrame
                    for source_id, target_ids in cross_refs.items():
                        # Track this source as visited
                        if source_id not in visited[source_db]:
                            visited[source_db].append(source_id)

                        # Add relations to DataFrame
                        for target_id in target_ids:
                            new_row = pd.DataFrame(
                                [
                                    {
                                        "source_id": source_id,
                                        "target_id": target_id,
                                        "source_db": source_db,
                                        "target_db": target_db,
                                    }
                                ]
                            )
                            nonlocal relations_df
                            relations_df = pd.concat(
                                [relations_df, new_row], ignore_index=True
                            )

                    # Prepare entities for next recursion
                    next_entities = [
                        (target_id, target_db)
                        for target_id, target_ids in cross_refs.items()
                        for target_id in target_ids
                        if target_id not in visited.get(target_db, [])
                    ]

                    if next_entities:
                        recursive_query(next_entities, current_depth + 1)

        # Start recursion
        recursive_query(entities, 0)

        return relations_df

    def search_and_retrieve(
        self, query: str, max_studies: Optional[int] = None
    ) -> List[Dict[str, Any]]:
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
            ...     print(f"Study: {study.get('acc')} - ...")
            ...     print(f"  Samples: {len(study.get('samples', []))}")
        """
        # Step 1: Find all relevant study accessions
        study_accessions = self.search_multi_level(query, max_studies)

        if not study_accessions:
            return []

        # Step 2: Get full nested data for those studies
        return self.get_full_study_data(study_accessions)
