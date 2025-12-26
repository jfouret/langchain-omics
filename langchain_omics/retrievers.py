"""SRA retrievers using EBI Search API."""

import json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .utils.unified_sra_client import UnifiedSRAClient


class SRARetriever(BaseRetriever):
    """SRA retriever using EBI Search API with multiple search modes."""

    database: str = "sra-sample"
    search_mode: str = "domain"
    k: int = 10
    max_studies: Optional[int] = None
    append_cross_ref: bool = True
    fetch_details: bool = False

    def __init__(self, **kwargs):
        """Initialize SRA retriever with validation."""
        super().__init__(**kwargs)
        
        # Initialize UnifiedSRAClient instance using object.__setattr__ to bypass Pydantic
        object.__setattr__(self, '_client', UnifiedSRAClient())
        
        # Validate database
        valid_databases = ["sra-sample", "sra-study", "sra-experiment", "sra-run"]
        if self.database not in valid_databases:
            raise ValueError(
                f"Invalid database '{self.database}'. Must be one of: {valid_databases}"
            )
        
        # Validate search_mode
        valid_modes = ["domain", "multi-level", "full"]
        if self.search_mode not in valid_modes:
            raise ValueError(
                f"Invalid search_mode '{self.search_mode}'. Must be one of: {valid_modes}"
            )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Retrieve relevant SRA documents for given query.
        
        Args:
            query: Search query string
            run_manager: Callback manager for retriever run
            **kwargs: Additional keyword arguments
            
        Returns:
            List of relevant Document objects
        """
        k = kwargs.get("k", self.k)
        search_mode = kwargs.get("search_mode", self.search_mode)
        max_studies = kwargs.get("max_studies", self.max_studies)
        append_cross_ref = kwargs.get("append_cross_ref", self.append_cross_ref)
        fetch_details = kwargs.get("fetch_details", self.fetch_details)
        
        try:
            if search_mode == "domain":
                return self._search_domain_mode(
                    query=query, k=k, append_cross_ref=append_cross_ref,
                    fetch_details=fetch_details, run_manager=run_manager
                )
            elif search_mode == "multi-level":
                return self._search_multilevel_mode(
                    query=query, k=k, max_studies=max_studies,
                    append_cross_ref=append_cross_ref, fetch_details=fetch_details,
                    run_manager=run_manager
                )
            elif search_mode == "full":
                return self._search_full_mode(
                    query=query, k=k, max_studies=max_studies, run_manager=run_manager
                )
            else:
                raise ValueError(f"Unknown search_mode: {search_mode}")
        except Exception as e:
            run_manager.on_text(f"Error retrieving SRA documents: {e}")
            return []

    def _search_domain_mode(
        self, query: str, k: int, append_cross_ref: bool,
        fetch_details: bool, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Domain mode: simple search within specified database.
        
        Args:
            query: Search query string
            k: Number of documents to return
            append_cross_ref: Whether to include cross-references
            fetch_details: Whether to fetch related entity details
            run_manager: Callback manager
            
        Returns:
            List of Document objects
        """
        # Use UnifiedSRAClient.search_domain() to find entries
        entries = self._client.search_domain(self.database, query, size=k)
        
        if not entries:
            return []
        
        # Extract entry IDs
        entry_ids = [entry.get("acc") or entry.get("id") for entry in entries]
        entry_ids = [eid for eid in entry_ids if eid]
        
        if not entry_ids:
            return []
        
        # Use UnifiedSRAClient.get_entries_batch() to get full entry data
        full_entries = self._client.get_entries_batch(self.database, entry_ids)
        
        # Convert each entry to Document with JSON content and metadata
        documents = []
        for entry in full_entries:
            try:
                doc = self._entry_to_document(
                    entry=entry, database=self.database,
                    append_cross_ref=append_cross_ref, fetch_details=fetch_details
                )
                documents.append(doc)
            except Exception as e:
                run_manager.on_text(f"Error processing entry: {e}")
                continue
        
        # Return exactly k documents
        return documents[:k]

    def _search_multilevel_mode(
        self, query: str, k: int, max_studies: Optional[int],
        append_cross_ref: bool, fetch_details: bool,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Multi-level mode: search across domains using cross-references.
        
        Args:
            query: Search query string
            k: Number of documents to return
            max_studies: Maximum number of studies
            append_cross_ref: Whether to include cross-references
            fetch_details: Whether to fetch related entity details
            run_manager: Callback manager
            
        Returns:
            List of Document objects
        """
        # Use UnifiedSRAClient.search_multi_level() to find studies
        study_accessions = self._client.search_multi_level(query, max_studies)
        
        if not study_accessions:
            return []
        
        # Use UnifiedSRAClient.get_entries_batch() to get study data
        studies = self._client.get_entries_batch("sra-study", study_accessions[:k])
        
        # Convert each study to Document with JSON content and metadata
        documents = []
        for study in studies:
            try:
                doc = self._entry_to_document(
                    entry=study, database="sra-study",
                    append_cross_ref=append_cross_ref, fetch_details=fetch_details
                )
                documents.append(doc)
            except Exception as e:
                run_manager.on_text(f"Error processing study: {e}")
                continue
        
        return documents[:k]

    def _search_full_mode(
        self, query: str, k: int, max_studies: Optional[int],
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Full mode: retrieve complete nested study data.
        
        Args:
            query: Search query string
            k: Number of documents to return
            max_studies: Maximum number of studies
            run_manager: Callback manager
            
        Returns:
            List of Document objects
        """
        # Use UnifiedSRAClient.search_and_retrieve() to get nested study data
        studies = self._client.search_and_retrieve(query, max_studies)
        
        if not studies:
            return []
        
        # Convert each study to Document with JSON content and metadata
        documents = []
        for study in studies[:k]:
            try:
                doc = self._study_to_document(study=study)
                documents.append(doc)
            except Exception as e:
                run_manager.on_text(f"Error processing study: {e}")
                continue
        
        return documents

    def _entry_to_document(
        self, entry: Dict[str, Any], database: str,
        append_cross_ref: bool, fetch_details: bool
    ) -> Document:
        """Convert entry to Document with JSON content and metadata.
        
        Args:
            entry: Entry dictionary
            database: SRA database name
            append_cross_ref: Whether to include cross-references
            fetch_details: Whether to fetch related entity details
            
        Returns:
            LangChain Document object
        """
        entry_id = entry.get("acc") or entry.get("id")
        
        # Build page content as JSON structure
        page_content_dict = {
            "id": entry_id,
            "type": database.replace("sra-", "").capitalize(),
            "database": database,
            "fields": {}
        }
        
        # Add all non-None fields
        for key, value in entry.items():
            if key not in ["acc", "id", "source"] and value is not None:
                page_content_dict["fields"][key] = value
        
        page_content = json.dumps(page_content_dict, indent=2)
        
        # Build metadata
        metadata = {
            "primary_id": entry_id,
            "database": database,
            "source": "EBI Search",
            "url": f"https://www.ebi.ac.uk/ena/browser/view/{entry_id}"
        }
        
        return Document(page_content=page_content, metadata=metadata)

    def _study_to_document(self, study: Dict[str, Any]) -> Document:
        """Convert nested study to Document with JSON content.
        
        Args:
            study: Nested study dictionary
            
        Returns:
            LangChain Document object
        """
        study_id = study.get("acc") or study.get("id")
        
        # Build simplified page content
        page_content_dict = {
            "id": study_id,
            "type": "Study",
            "database": "sra-study",
            "fields": {
                "title": study.get("title", ""),
                "description": study.get("description", "")
            },
            "experiments": []
        }
        
        # Add experiments
        for exp in study.get("experiments", []):
            exp_dict = {
                "id": exp.get("acc") or exp.get("id"),
                "title": exp.get("title", ""),
                "platform": exp.get("platform", "")
            }
            page_content_dict["experiments"].append(exp_dict)
        
        page_content = json.dumps(page_content_dict, indent=2)
        
        # Build metadata
        metadata = {
            "primary_id": study_id,
            "database": "sra-study",
            "source": "EBI Search",
            "url": f"https://www.ebi.ac.uk/ena/browser/view/{study_id}",
            "num_experiments": len(study.get("experiments", []))
        }
        
        return Document(page_content=page_content, metadata=metadata)

