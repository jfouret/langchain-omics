"""SRA retrievers using EBI Search API."""

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .utils.ebi_search_client import EBISearchAPIWrapper


# SRA Database Configuration
SRA_DATABASES = {
    "sra-sample": {
        "name": "Sample",
        "key_fields": ["acc", "id", "description", "scientific_name", "country", 
                      "region", "collection_date", "center_name", "strain", "host"],
        "cross_ref_fields": ["BIOSAMPLE", "TAXON"],
        "cross_ref_targets": {
            "BIOSAMPLE": "biosamples",
            "TAXON": "taxonomy"
        },
        "related_domains": ["sra-study", "sra-experiment"]
    },
    "sra-study": {
        "name": "Study",
        "key_fields": ["acc", "id", "title", "description", "center_name", 
                      "first_public_date"],
        "related_domains": ["sra-sample", "sra-experiment", "sra-project"]
    },
    "sra-experiment": {
        "name": "Experiment", 
        "key_fields": ["acc", "id", "title", "description", "platform",
                      "instrument_model", "library_strategy"],
        "related_domains": ["sra-sample", "sra-study", "sra-run"]
    },
    "sra-run": {
        "name": "Run",
        "key_fields": ["acc", "id", "description", "total_spots", "total_bases"],
        "related_domains": ["sra-experiment"]
    }
}


class SRARetriever(BaseRetriever, EBISearchAPIWrapper):
    """SRA retriever using EBI Search API.
    
    This retriever searches the Sequence Read Archive (SRA) databases using the
    EBI Search API. It supports searching across different SRA entity types
    (samples, studies, experiments, runs) and can retrieve cross-references
    and related entities.
    
    Setup:
        Install ``langchain-omics`` and optionally set environment variables
        for API configuration.

        .. code-block:: bash

            pip install -U langchain-omics

    Key init args:
        database: str
            SRA database to search ('sra-sample', 'sra-study', 'sra-experiment', 'sra-run')
        append_cross_ref: bool
            Whether to append cross-reference info to document text (default: True)
        fetch_details: bool
            Whether to make additional API calls for details (default: False)
        k: int
            Number of documents to return (default: 10)

    Instantiate:
        .. code-block:: python

            from langchain_omics import SRARetriever

            # Search SRA samples (fast mode)
            retriever = SRARetriever(
                database="sra-sample",
                append_cross_ref=True,
                fetch_details=False,
                k=5
            )

    Usage:
        .. code-block:: python

            query = "human cancer RNA-seq"
            docs = retriever.invoke(query)

        .. code-block:: none

            [Document(page_content='{
              "id": "SAMN12345678",
              "type": "Sample",
              "database": "sra-sample",
              "fields": {
                "description": "Human cancer tissue RNA-seq sample",
                "scientific_name": "Homo sapiens",
                "country": "United States"
              },
              "cross_references": {
                "BIOSAMPLE": ["SAMN12345678"],
                "TAXON": ["9606"]
              }
            }', metadata={...})]

    Use within a chain:
        .. code-block:: python

            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            chain.invoke("What cancer types are available in the SRA?")

        .. code-block:: none

             Based on the SRA samples retrieved, several cancer types are available
             including breast cancer, lung cancer, and colorectal cancer samples
             from human subjects.
    """

    database: str = "sra-sample"  # sra-sample, sra-study, sra-experiment, sra-run
    append_cross_ref: bool = True  # Whether to append cross-ref info to text
    fetch_details: bool = False  # Whether to make additional API calls for details
    k: int = 10  # Number of documents to return
    ignore_k_error: bool = False  # Whether to ignore k > max_k errors

    def __init__(self, **kwargs):
        """Initialize SRA retriever with validation."""
        super().__init__(**kwargs)
        
        # Validate database
        if self.database not in SRA_DATABASES:
            raise ValueError(
                f"Invalid database '{self.database}'. "
                f"Must be one of: {list(SRA_DATABASES.keys())}"
            )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Retrieve relevant SRA documents for the given query.
        
        Args:
            query: Search query string
            run_manager: Callback manager for the retriever run
            **kwargs: Additional keyword arguments
            
        Returns:
            List of relevant Document objects
        """
        # Determine number of results to fetch
        k = kwargs.get("k", self.k)
        ignore_k_error = kwargs.get("ignore_k_error", self.ignore_k_error)
        fetch_details = kwargs.get("fetch_details", self.fetch_details)
        
        # Validate k against max_k unless ignore_k_error is True
        max_k = self.get_max_k(query)
        if k > max_k:
            if ignore_k_error:
                size = max_k
            else:
                raise ValueError(
                    f"Requested k={k} exceeds maximum available results ({max_k}) "
                    f"for query '{query}' in database '{self.database}'. "
                    f"Set ignore_k_error=True to allow this."
                )
        else:
            size = k
        
        # Get domain configuration
        domain_config = SRA_DATABASES[self.database]
        
        # Get fields to retrieve (always include cross-ref fields)
        fields = domain_config.get("key_fields", [])
        if "cross_ref_fields" in domain_config:
            fields.extend(domain_config["cross_ref_fields"])
        
        try:
            # Use paginated loading for efficient API usage
            if size > 25:
                entries = self._load_paginated(
                    query=query,
                    domain=self.database,
                    total_needed=size,
                    fields=fields
                )
            else:
                entries = self._load(
                    query=query,
                    domain=self.database,
                    size=size,
                    fields=fields
                )
            
            # Convert entries to documents
            documents = []
            for entry in entries:
                try:
                    doc = self._dict2document(
                        entry=entry,
                        domain=self.database,
                        append_cross_ref=self.append_cross_ref,
                        fetch_details=fetch_details,
                        domain_config=domain_config
                    )
                    documents.append(doc)
                except Exception as e:
                    # Log error but continue with other entries
                    run_manager.on_text(f"Error processing entry: {e}")
                    continue
            
            # Return exact number if k is specified
            return documents[:k]
            
        except Exception as e:
            run_manager.on_text(f"Error retrieving SRA documents: {e}")
            return []

    def _get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database configuration.
        
        Returns:
            Dictionary with database configuration information
        """
        return SRA_DATABASES.get(self.database, {})

    def get_available_databases(self) -> List[str]:
        """Get list of available SRA databases.
        
        Returns:
            List of available database identifiers
        """
        return list(SRA_DATABASES.keys())

    def switch_database(self, database: str) -> None:
        """Switch to a different SRA database.
        
        Args:
            database: Database identifier to switch to
            
        Raises:
            ValueError: If database is not valid
        """
        if database not in SRA_DATABASES:
            raise ValueError(
                f"Invalid database '{database}'. "
                f"Must be one of: {list(SRA_DATABASES.keys())}"
            )
        self.database = database

    def get_max_k(self, query: str) -> int:
        """Get the maximum number of results available for a query.
        
        Args:
            query: Search query string
            
        Returns:
            Maximum number of results available for the query in the current database
        """
        return super().get_max_k(query, self.database)
