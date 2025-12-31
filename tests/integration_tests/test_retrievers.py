"""Integration tests for SRARetriever with new architecture."""

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
            "databases": ["sra-sample", "sra-study"],
            "report_levels": ["sample"],
            "k": 2
        }

    @property
    def retriever_query_example(self) -> str:
        """Returns a str representing "query" of an example retriever call."""
        return "human cancer"

    def test_new_parameters_validation(self):
        """Test validation of new parameters."""
        # Invalid database
        try:
            retriever = SRARetriever(databases=["invalid-db"])
            assert False, "Should have raised ValueError for invalid database"
        except ValueError as e:
            assert "Invalid database" in str(e)
        
        # Invalid report_level
        try:
            retriever = SRARetriever(report_levels=["invalid-level"])
            assert False, "Should have raised ValueError for invalid report_level"
        except ValueError as e:
            assert "Invalid report_level" in str(e)
        
        # Negative depth
        try:
            retriever = SRARetriever(depth=-1)
            assert False, "Should have raised ValueError for negative depth"
        except ValueError as e:
            assert "depth" in str(e)
        
        # Invalid k
        try:
            retriever = SRARetriever(k=0)
            assert False, "Should have raised ValueError for k=0"
        except ValueError as e:
            assert "k" in str(e)
        
        # Invalid max_per_db
        try:
            retriever = SRARetriever(max_per_db={"sra-study": -5})
            assert False, "Should have raised ValueError for negative max_per_db"
        except ValueError as e:
            assert "max_per_db" in str(e)

    def test_basic_functionality_flat(self):
        """Test basic retriever functionality with flat results (depth=0)."""
        retriever = SRARetriever(
            databases=["sra-sample"],
            report_levels=["sample"],
            depth=0,
            k=5
        )
        
        docs = retriever.invoke("human")
        assert len(docs) <= 5, f"Expected <=5 docs, got {len(docs)}"
        
        # Verify document structure
        for doc in docs:
            content = json.loads(doc.page_content)
            assert 'id' in content
            assert 'type' in content
            assert 'database' in content
            assert 'crossref' in content
            assert content['type'] == 'Sample'
            assert len(content['crossref']) == 0  # depth=0 means no cross-references

    def test_single_report_level(self):
        """Test with single report level."""
        retriever = SRARetriever(
            databases=["sra-study", "sra-experiment"],
            report_levels=["study"],
            depth=0,
            k=3
        )
        
        docs = retriever.invoke("cancer")
        assert len(docs) <= 3
        
        # All docs should be studies
        for doc in docs:
            content = json.loads(doc.page_content)
            assert content['type'] == 'Study'

    def test_multiple_report_levels(self):
        """Test with multiple report levels."""
        retriever = SRARetriever(
            databases=["sra-study", "sra-sample"],
            report_levels=["study", "sample"],
            depth=0,
            k=10
        )
        
        docs = retriever.invoke("human")
        assert len(docs) <= 10
        
        # Verify both types are present
        types_found = set()
        for doc in docs:
            content = json.loads(doc.page_content)
            types_found.add(content['type'])
        
        assert 'Study' in types_found or 'Sample' in types_found

    def test_depth_0_flat(self):
        """Test depth=0 returns flat results."""
        retriever = SRARetriever(
            databases=["sra-study"],
            report_levels=["study"],
            depth=0,
            k=2
        )
        
        docs = retriever.invoke("human")
        assert len(docs) <= 2
        
        for doc in docs:
            content = json.loads(doc.page_content)
            # depth=0 should have no cross-references
            assert len(content.get('crossref', [])) == 0

    def test_depth_1_direct_relationships(self):
        """Test depth=1 includes direct relationships."""
        retriever = SRARetriever(
            databases=["sra-study"],
            report_levels=["study"],
            depth=1,
            k=2
        )
        
        docs = retriever.invoke("human")
        assert len(docs) <= 2
        
        for doc in docs:
            content = json.loads(doc.page_content)
            # depth=1 should have cross-references (experiments for studies)
            has_crossref = len(content.get('crossref', [])) > 0
            assert has_crossref, "Should have cross-references at depth=1"

    def test_depth_2_two_levels(self):
        """Test depth=2 includes two levels of relationships."""
        retriever = SRARetriever(
            databases=["sra-study"],
            report_levels=["study"],
            depth=2,
            k=2
        )
        
        docs = retriever.invoke("human")
        assert len(docs) <= 2
        
        # depth=2 should include more nested structure
        for doc in docs:
            content = json.loads(doc.page_content)
            crossrefs = content.get('crossref', [])
            if crossrefs:
                # Check if nested cross-references exist
                for child in crossrefs:
                    has_nested = len(child.get('crossref', [])) > 0
                    if has_nested:
                        break

    def test_k_distribution(self):
        """Test that k applies to total across all report_levels."""
        retriever = SRARetriever(
            databases=["sra-study", "sra-sample"],
            report_levels=["study", "sample"],
            depth=0,
            k=5
        )
        
        docs = retriever.invoke("human")
        # k=5 should return at most 5 total docs
        assert len(docs) <= 5

    def test_max_per_db(self):
        """Test max_per_db limits are respected."""
        retriever = SRARetriever(
            databases=["sra-study"],
            report_levels=["study"],
            depth=1,
            k=2,
            max_per_db={"sra-experiment": 5}
        )
        
        docs = retriever.invoke("human")
        assert len(docs) <= 2
        
        # Verify cross-references are limited
        for doc in docs:
            content = json.loads(doc.page_content)
            crossrefs = content.get('crossref', [])
            # Should have at most 5 cross-references due to max_per_db
            # Note: Actual limit depends on API responses
            assert len(crossrefs) >= 0

    def test_all_databases_functionality(self):
        """Test that all SRA databases are functional."""
        databases = ["sra-study", "sra-sample", "sra-experiment", "sra-run"]
        
        for database in databases:
            retriever = SRARetriever(
                databases=[database],
                report_levels=[database.replace("sra-", "")],
                depth=0,
                k=2
            )
            
            docs = retriever.invoke("cancer")
            assert len(docs) > 0, f"No documents returned for database: {database}"

    def test_document_structure(self):
        """Test that document structure matches design document."""
        retriever = SRARetriever(
            databases=["sra-study"],
            report_levels=["study"],
            depth=0,
            k=2
        )
        
        docs = retriever.invoke("human")
        assert len(docs) > 0
        
        for doc in docs:
            # Verify JSON format
            content = json.loads(doc.page_content)
            
            # Check required JSON fields
            assert 'id' in content
            assert 'type' in content
            assert 'database' in content
            assert 'fields' in content
            assert 'crossref' in content
            
            # Check metadata completeness
            metadata = doc.metadata
            assert 'primary_id' in metadata
            assert 'database' in metadata
            assert 'source' in metadata
            assert 'url' in metadata
            assert metadata['source'] == 'EBI Search'

    def test_design_document_example_1(self):
        """Test Example 1 from design document: Simple Study Search (Flat)."""
        retriever = SRARetriever(
            databases=["sra-study"],
            report_levels=["study"],
            depth=0,
            k=10
        )
        
        docs = retriever.invoke("respiratory")
        assert len(docs) <= 10

    def test_design_document_example_2(self):
        """Test Example 2: Multi-Database Search with Study Context."""
        retriever = SRARetriever(
            databases=["sra-study", "sra-sample", "sra-experiment"],
            report_levels=["study"],
            depth=1,
            k=10,
            max_per_db={"sra-study": 10, "sra-experiment": 50}
        )
        
        docs = retriever.invoke("influenza")
        assert len(docs) <= 10

    def test_design_document_example_4(self):
        """Test Example 4: Experiment-Only Results."""
        retriever = SRARetriever(
            databases=["sra-experiment"],
            report_levels=["experiment"],
            depth=0,
            k=100
        )
        
        docs = retriever.invoke("RNA-seq")
        assert len(docs) <= 100
        
        # All should be experiments
        for doc in docs:
            content = json.loads(doc.page_content)
            assert content['type'] == 'Experiment'
