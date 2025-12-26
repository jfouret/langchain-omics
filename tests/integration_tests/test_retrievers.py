"""Integration tests for SRARetriever."""

import json
from typing import Type

from langchain_omics.retrievers import SRARetriever
from langchain_tests.integration_tests import RetrieversIntegrationTests


class TestSRARetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[SRARetriever]:
        """Get an empty retriever for integration tests."""
        return SRARetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {
            "database": "sra-sample",
            "search_mode": "domain",
            "k": 2,
            "fetch_details": False
        }

    @property
    def retriever_query_example(self) -> str:
        """Returns a str representing "query" of an example retriever call."""
        return "human cancer"

    def test_basic_functionality(self):
        """Test basic retriever functionality with domain mode."""
        databases = ["sra-sample", "sra-study", "sra-experiment", "sra-run"]
        query = "human"
        
        for database in databases:
            retriever = SRARetriever(database=database, k=5, search_mode="domain")
            
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
            # Test basic functionality with each database
            retriever = SRARetriever(database=database, k=5, search_mode="domain")
            docs = retriever.invoke(query)
            assert len(docs) > 0, f"No documents returned for database: {database}"

    def test_search_modes(self):
        """Test all three search modes."""
        query = "human cancer"
        
        # Domain mode
        retriever_domain = SRARetriever(
            database="sra-sample",
            search_mode="domain",
            k=3
        )
        docs_domain = retriever_domain.invoke(query)
        assert len(docs_domain) > 0, "Domain mode returned no documents"
        
        # Multi-level mode
        retriever_multilevel = SRARetriever(
            database="sra-study",
            search_mode="multi-level",
            k=3,
            max_studies=10
        )
        docs_multilevel = retriever_multilevel.invoke(query)
        assert len(docs_multilevel) > 0, "Multi-level mode returned no documents"
        
        # Full mode
        retriever_full = SRARetriever(
            database="sra-study",
            search_mode="full",
            k=2,
            max_studies=5
        )
        docs_full = retriever_full.invoke(query)
        assert len(docs_full) > 0, "Full mode returned no documents"
        
        # Full mode should have experiments in content
        if docs_full:
            content_data = json.loads(docs_full[0].page_content)
            assert 'experiments' in content_data, "Full mode missing experiments in content"

    def test_full_mode_structure(self):
        """Test that full mode returns proper nested study structure."""
        retriever = SRARetriever(
            database="sra-study",
            search_mode="full",
            k=2,
            max_studies=3
        )
        
        docs = retriever.invoke("human")
        assert len(docs) > 0, "No documents returned in full mode"
        
        # Verify nested structure
        for doc in docs:
            content_data = json.loads(doc.page_content)
            assert 'experiments' in content_data, "Missing experiments in full mode content"
            assert len(content_data['experiments']) > 0, "No experiments found in full mode"
            
            # Check experiment structure
            exp = content_data['experiments'][0]
            assert 'id' in exp, "Experiment missing id"
            assert 'title' in exp, "Experiment missing title"
            
            # Check metadata
            assert 'num_experiments' in doc.metadata, "Missing num_experiments in metadata"
            assert doc.metadata['num_experiments'] > 0, "num_experiments should be > 0"

    def test_json_content_format(self):
        """Test that document content is valid JSON with expected structure."""
        retriever = SRARetriever(database="sra-sample", k=2, search_mode="domain")
        docs = retriever.invoke("human cancer")
        assert len(docs) > 0, "No documents returned"
        
        for doc in docs:
            # Verify JSON format
            try:
                content_data = json.loads(doc.page_content)
            except json.JSONDecodeError:
                assert False, "Page content is not valid JSON"
            
            # Check required JSON fields
            required_fields = ['id', 'type', 'database', 'fields']
            for field in required_fields:
                assert field in content_data, f"Missing {field} in JSON content"
            
            # Check metadata completeness
            metadata = doc.metadata
            required_metadata = ['primary_id', 'database', 'source', 'url']
            for field in required_metadata:
                assert field in metadata, f"Missing {field} in metadata"

    def test_k_parameter(self):
        """Test that k parameter is respected."""
        query = "human"
        
        # Test small k
        retriever_small = SRARetriever(database="sra-sample", k=5, search_mode="domain")
        docs_small = retriever_small.invoke(query)
        assert len(docs_small) <= 5, f"Expected <=5 docs, got {len(docs_small)}"
        
        # Test larger k
        retriever_large = SRARetriever(database="sra-sample", k=20, search_mode="domain")
        docs_large = retriever_large.invoke(query)
        assert len(docs_large) <= 20, f"Expected <=20 docs, got {len(docs_large)}"

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Invalid database
        try:
            retriever = SRARetriever(database="invalid-db")
            assert False, "Should have raised ValueError for invalid database"
        except ValueError as e:
            assert "Invalid database" in str(e)
        
        # Invalid search_mode
        try:
            retriever = SRARetriever(search_mode="invalid-mode")
            assert False, "Should have raised ValueError for invalid search_mode"
        except ValueError as e:
            assert "Invalid search_mode" in str(e)
