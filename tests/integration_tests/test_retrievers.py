from typing import Type
import json

from langchain_omics.retrievers import SRARetriever
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)

class TestSRARetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[SRARetriever]:
        """Get an empty retriever for integration tests."""
        return SRARetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {
            "database": "sra-sample",
            "k": 2,
            "fetch_details": False  # Fast mode for tests
        }

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "human cancer"

    def test_basic_functionality(self):
        """Test basic retriever functionality across databases."""
        databases = ["sra-sample", "sra-study", "sra-experiment", "sra-run"]
        query = "human"
        
        for database in databases:
            retriever = SRARetriever(
                database=database,
                k=5,
                fetch_details=False
            )
            
            # Test that retriever can retrieve documents
            docs = retriever.invoke(query)
            assert len(docs) > 0, f"Retriever returned no documents for {database}"
            assert len(docs) <= 5, f"K parameter not respected for {database}"
            
            # Verify document structure
            first_doc = docs[0]
            assert hasattr(first_doc, 'page_content'), "Document missing page_content"
            assert hasattr(first_doc, 'metadata'), "Document missing metadata"
            assert first_doc.metadata.get('database') == database, "Incorrect database in metadata"
            assert first_doc.metadata.get('primary_id'), "Missing primary_id in metadata"
            
            # Verify JSON content format
            try:
                content_data = json.loads(first_doc.page_content)
                assert 'id' in content_data, "Missing id in JSON content"
                assert 'type' in content_data, "Missing type in JSON content"
                assert 'database' in content_data, "Missing database in JSON content"
                assert 'fields' in content_data, "Missing fields in JSON content"
            except json.JSONDecodeError:
                assert False, f"Page content is not valid JSON for {database}"

    def test_all_databases_functionality(self):
        """Test that all SRA databases are functional and return results."""
        databases = ["sra-sample", "sra-study", "sra-experiment", "sra-run"]
        query = "cancer"
        
        for database in databases:
            retriever = SRARetriever(
                database=database,
                k=5,
                fetch_details=False
            )
            
            # Test basic functionality
            docs = retriever.invoke(query)
            assert len(docs) > 0, f"No documents returned for database: {database}"
            
            # Test database switching
            original_db = retriever.database
            retriever.switch_database(database)
            assert retriever.database == database
            
            # Test available databases
            available = retriever.get_available_databases()
            assert database in available, f"Database {database} not in available list"
            
            # Test database info
            db_info = retriever._get_database_info()
            assert db_info.get('name'), f"No name found for database {database}"
            assert 'key_fields' in db_info, f"No key_fields found for database {database}"

    def test_cross_references(self):
        """Test cross-reference functionality."""
        retriever = SRARetriever(
            database="sra-sample",
            k=3,
            fetch_details=False  # Fast mode
        )
        
        docs = retriever.invoke("human")
        assert len(docs) > 0, "No documents returned for cross-reference test"
        
        # Check that cross-references are included in metadata
        first_doc = docs[0]
        metadata = first_doc.metadata
        
        assert 'cross_references' in metadata, "Missing cross_references in metadata"
        assert 'related_entities' in metadata, "Missing related_entities in metadata"
        
        # Verify JSON content includes cross-references
        content_data = json.loads(first_doc.page_content)
        assert 'cross_references' in content_data, "Missing cross_references in JSON content"

    def test_pagination_performance(self):
        """Test pagination functionality with larger k values."""
        query = "human"
        
        # Test small k (single request)
        retriever_small = SRARetriever(database="sra-sample", k=5, fetch_details=False)
        docs_small = retriever_small.invoke(query)
        assert len(docs_small) == 5, f"Expected 5 docs, got {len(docs_small)}"
        
        # Test larger k (pagination)
        retriever_large = SRARetriever(database="sra-sample", k=50, fetch_details=False)
        docs_large = retriever_large.invoke(query)
        assert len(docs_large) == 50, f"Expected 50 docs, got {len(docs_large)}"

    def test_fetch_details_mode(self):
        """Test detailed vs fast mode."""
        query = "human"
        
        # Fast mode
        retriever_fast = SRARetriever(database="sra-sample", k=2, fetch_details=False)
        docs_fast = retriever_fast.invoke(query)
        
        # Detailed mode
        retriever_detailed = SRARetriever(database="sra-sample", k=2, fetch_details=True)
        docs_detailed = retriever_detailed.invoke(query)
        
        assert len(docs_fast) == len(docs_detailed), "Different number of docs returned"
        
        # Detailed mode should have additional metadata
        fast_metadata = docs_fast[0].metadata
        detailed_metadata = docs_detailed[0].metadata
        
        # Both should have basic cross_references
        assert 'cross_references' in fast_metadata
        assert 'cross_references' in detailed_metadata
        
        # Only detailed should have cross_reference_details
        assert 'cross_reference_details' not in fast_metadata
        # Note: cross_reference_details might be empty if no details are available

    def test_json_content_format(self):
        """Test that document content is valid JSON with expected structure."""
        retriever = SRARetriever(database="sra-sample", k=2, fetch_details=False)
        docs = retriever.invoke("human cancer")
        assert len(docs) > 0, "No documents returned"
        
        for doc in docs:
            # Verify JSON format
            try:
                content_data = json.loads(doc.page_content)
            except json.JSONDecodeError:
                assert False, "Page content is not valid JSON"
            
            # Check required JSON fields
            required_fields = ['id', 'type', 'database', 'fields', 'cross_references']
            for field in required_fields:
                assert field in content_data, f"Missing {field} in JSON content"
            
            # Check metadata completeness
            metadata = doc.metadata
            required_metadata = ['primary_id', 'database', 'entity_type', 'source', 'url']
            for field in required_metadata:
                assert field in metadata, f"Missing {field} in metadata"
