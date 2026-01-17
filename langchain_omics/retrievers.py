"""SRA retrievers using EBI Search API with separated search and reporting phases."""

import json
from typing import Any, ClassVar, Dict, List, Set, Tuple

import pandas as pd
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from .utils.unified_sra_client import UnifiedSRAClient


class SRARetriever(BaseRetriever):
    """SRA retriever using EBI Search API with separated search and reporting phases.

    This retriever separates concerns into two phases:
    1. Search Phase: Discover relevant entities across databases
    2. Reporting Phase: Build hierarchies and format output

    This provides flexible control over both discovery scope and output format.
    """

    databases: List[str] = ["sra-study", "sra-sample", "sra-experiment", "sra-run"]
    k: int = 10
    report_levels: List[str] = ["study"]
    depth: int = 0
    report_only_discovered: bool = True
    max_per_db: Dict[str, int] = {}
    depth_k_limits: Dict[str, int] = {}

    # Cross-reference mapping based on EBI schema (undirected relationships)
    CROSS_REF_MAP: ClassVar[Dict[str, List[str]]] = {
        "sra-study": ["sra-experiment"],
        "sra-experiment": ["sra-study", "sra-sample", "sra-run"],
        "sra-sample": ["sra-experiment"],
        "sra-run": ["sra-experiment"],
    }

    def __init__(self, **kwargs):
        """Initialize SRA retriever with validation."""
        super().__init__(**kwargs)

        # Initialize UnifiedSRAClient instance using object.__setattr__ to bypass Pydantic
        object.__setattr__(self, "_client", UnifiedSRAClient())

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate all retriever parameters."""
        # Validate databases
        self._validate_databases()

        # Validate report_levels
        self._validate_report_levels()

        # Validate depth
        if not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError(f"depth must be a non-negative integer, got: {self.depth}")

        # Validate k
        if not isinstance(self.k, int) or self.k <= 0:
            raise ValueError(f"k must be a positive integer, got: {self.k}")

        # Validate max_per_db values
        for db, limit in self.max_per_db.items():
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError(
                    f"max_per_db['{db}'] must be a positive integer, got: {limit}"
                )

        # Validate depth_k_limits values
        for db, limit in self.depth_k_limits.items():
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError(
                    f"depth_k_limits['{db}'] must be a positive integer, got: {limit}"
                )

    def _validate_databases(self) -> None:
        """Validate all databases in list are valid."""
        valid_databases = ["sra-sample", "sra-study", "sra-experiment", "sra-run"]
        for database in self.databases:
            if database not in valid_databases:
                raise ValueError(
                    f"Invalid database '{database}'. Must be one of: {valid_databases}"
                )

    def _validate_report_levels(self) -> None:
        """Validate all report_levels are valid entity types."""
        valid_levels = ["study", "sample", "experiment", "run"]
        for level in self.report_levels:
            if level not in valid_levels:
                raise ValueError(
                    f"Invalid report_level '{level}'. Must be one of: {valid_levels}"
                )

    def _entity_type_to_db(self, entity_type: str) -> str:
        """Map entity type to database name."""
        return f"sra-{entity_type}"

    def _db_to_entity_type(self, database: str) -> str:
        """Map database name to entity type."""
        return database.replace("sra-", "")

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Retrieve relevant SRA documents for given query.

        This implements the four-phase workflow:
        1. Discovery: Find all relevant entities across databases
        2. Selection: Choose k primary entities based on report_levels
        3. Hierarchy Building: Build hierarchical structures with depth
        4. Document Conversion: Convert to LangChain Documents

        Args:
            query: Search query string
            **kwargs: Additional keyword arguments

        Returns:
            List of relevant Document objects
        """
        try:
            # Extract parameters from kwargs or use defaults
            databases = kwargs.get("databases", self.databases)
            report_levels = kwargs.get("report_levels", self.report_levels)
            depth = kwargs.get("depth", self.depth)
            report_only_discovered = kwargs.get(
                "report_only_discovered", self.report_only_discovered
            )

            # Phase 1: Discovery - Find all relevant entities across databases
            discovered_ids = self._discover_entities(query, **kwargs)

            # Convert to set for O(1) lookup
            discovered_ids_set = set(discovered_ids)

            # Phase 2: Selection - Choose report entities and build relations
            report_ids, relations_df = self._select_report_entities(
                discovered_ids, report_levels, databases
            )

            # Phase 3 & 4: Hierarchy Building & Document Conversion
            return self._to_documents(
                report_ids,
                relations_df,
                discovered_ids_set,
                report_only_discovered,
                depth,
            )

        except Exception:
            return []

    def _discover_entities(self, query: str, **kwargs: Any) -> List[Tuple[str, str]]:
        """Phase 1: Search all databases and discover entities.

        Args:
            query: Search query string
            **kwargs: Additional keyword arguments including k, databases, max_per_db

        Returns:
            List of (ID, DATABASE) tuples for all discovered entities
        """

        discovered_ids = []
        k = kwargs.get("k", self.k)
        databases = kwargs.get("databases", self.databases)
        max_per_db = kwargs.get("max_per_db", self.max_per_db)

        # Step 1: Search all databases directly
        search_results = self._client.search_multiple_domains(
            databases, query, size=k, size_per_db=max_per_db
        )

        # Collect direct search results as tuples
        for database, entries in search_results.items():
            for entry in entries:
                entry_id = entry.get("acc") or entry.get("id")
                if entry_id:
                    discovered_ids.append((entry_id, database))

        return discovered_ids

    def _select_report_entities(
        self,
        discovered_ids: List[Tuple[str, str]],
        report_levels: List[str],
        databases: List[str],
    ) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
        """Phase 2: Select report entities and build cross-reference relations.

        Args:
            discovered_ids: List of (ID, DATABASE) tuples from Phase 1
            report_levels: List of entity types to report on (e.g., ["study"])
            databases: List of all databases being searched

        Returns:
            Tuple of (report_ids, relations_df) where:
            - report_ids: List of (ID, DATABASE) tuples matching report_levels
            - relations_df: DataFrame with columns [source_id, target_id, source_db, target_db]
        """
        # Build relations DataFrame using recursive cross-reference
        relations_df = self._client.fetch_crossref_recursively(
            discovered_ids, self.CROSS_REF_MAP, max_depth=2
        )

        # Convert report_levels to database names for matching
        report_dbs = [self._entity_type_to_db(level) for level in report_levels]

        # Extract report_ids (IDs where database matches report_levels)
        report_ids = [
            (entity_id, db) for entity_id, db in discovered_ids if db in report_dbs
        ]

        return report_ids, relations_df

    def _to_documents(
        self,
        report_ids: List[Tuple[str, str]],
        relations_df: pd.DataFrame,
        discovered_ids_set: Set[Tuple[str, str]],
        report_only_discovered: bool,
        depth: int,
    ) -> List[Document]:
        """Phases 3 & 4: Build hierarchy and convert to Documents.

        Args:
            report_ids: List of (ID, DATABASE) tuples for primary entities
            relations_df: DataFrame with cross-reference relations
            discovered_ids_set: Set of (ID, DATABASE) tuples for filtering
            report_only_discovered: If True, only include discovered entities
            depth: Number of levels to traverse in hierarchy

        Returns:
            List of Document objects with nested JSON content
        """
        if not report_ids:
            return []

        # Phase 3: Build hierarchical structures
        hierarchies = []

        for report_id, report_db in report_ids:
            # Build hierarchy for this report entity
            hierarchy = self._build_hierarchy_from_relations(
                report_id,
                report_db,
                relations_df,
                discovered_ids_set,
                report_only_discovered,
                depth,
                set(),  # visited set for cycle detection
            )
            hierarchies.append(hierarchy)

        # Collect all entity IDs that need field data
        all_entity_keys = self._collect_entity_keys(hierarchies)
        
        # Batch fetch entity data from API
        entity_data_map = self._fetch_entity_data(all_entity_keys)
        
        # Populate fields in hierarchies
        self._populate_hierarchy_fields(hierarchies, entity_data_map)

        # Phase 4: Convert to Documents
        documents = []

        for hierarchy in hierarchies:
            # Create Document with JSON content
            doc = Document(
                page_content=json.dumps(hierarchy, indent=2),
                metadata={
                    "id": hierarchy.get("id"),
                    "type": hierarchy.get("type"),
                    "database": hierarchy.get("database"),
                },
            )
            documents.append(doc)

        return documents

    def _build_hierarchy_from_relations(
        self,
        entity_id: str,
        entity_db: str,
        relations_df: pd.DataFrame,
        discovered_ids_set: Set[Tuple[str, str]],
        report_only_discovered: bool,
        depth: int,
        visited: Set[str],
    ) -> Dict[str, Any]:
        """Build hierarchical structure for a single entity using relations.

        Args:
            entity_id: Starting entity ID
            entity_db: Starting entity database
            relations_df: DataFrame with cross-reference relations
            discovered_ids_set: Set of (ID, DATABASE) tuples for filtering
            report_only_discovered: If True, only include discovered entities
            depth: Number of levels to traverse
            visited: Set of already-visited entity IDs

        Returns:
            Dictionary with hierarchical entity structure
        """
        # Mark as visited to prevent cycles
        visited.add(entity_id)

        # Get entity type from database
        entity_type = self._db_to_entity_type(entity_db)

        # Initialize hierarchy node (fields will be populated later)
        hierarchy = {
            "id": entity_id,
            "type": entity_type,
            "database": entity_db,
            "crossref": [],
        }

        # If depth <= 0, return without traversing cross-references
        if depth <= 0:
            return hierarchy

        # Find related entities from relations_df
        # Filter relations where source_id == entity_id and source_db == entity_db
        related = relations_df[
            (relations_df["source_id"] == entity_id)
            & (relations_df["source_db"] == entity_db)
        ]

        # Traverse to related entities
        for _, row in related.iterrows():
            target_id = row["target_id"]
            target_db = row["target_db"]
            target_key = (target_id, target_db)

            # Skip if already visited
            if target_id in visited:
                continue

            # Skip if report_only_discovered and target not in discovered set
            if report_only_discovered and target_key not in discovered_ids_set:
                continue

            # Recursively build child hierarchy
            child = self._build_hierarchy_from_relations(
                target_id,
                target_db,
                relations_df,
                discovered_ids_set,
                report_only_discovered,
                depth - 1,
                visited.copy(),  # Use copy to allow siblings to share same subtree
            )

            hierarchy["crossref"].append(child)

        return hierarchy

    def _collect_entity_keys(
        self, hierarchies: List[Dict[str, Any]]
    ) -> Set[Tuple[str, str]]:
        """Collect all entity (ID, DATABASE) tuples from hierarchies.
        Mandatory to fetch entries details per database type. 
        Note for later: see if we can use relations_df to avoid this step.

        Args:
            hierarchies: List of hierarchy dictionaries

        Returns:
            Set of (entity_id, entity_db) tuples
        """
        entity_keys = set()

        def collect_from_node(node: Dict[str, Any]) -> None:
            """Recursively collect entity keys from a node."""
            entity_id = node.get("id")
            entity_db = node.get("database")
            if entity_id and entity_db:
                entity_keys.add((entity_id, entity_db))

            # Recursively collect from cross-references
            for child in node.get("crossref", []):
                collect_from_node(child)

        for hierarchy in hierarchies:
            collect_from_node(hierarchy)

        return entity_keys

    def _fetch_entity_data(
        self, entity_keys: Set[Tuple[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Batch fetch entity data from API.

        Args:
            entity_keys: Set of (entity_id, entity_db) tuples

        Returns:
            Dictionary mapping entity_id to entity data
        """
        entries_across_dbs = {}

        # Group entity IDs by database for efficient batch fetching
        # XML API requires IDs in a single batch to be from the same database
        entities_by_db: Dict[str, List[str]] = {}
        for entity_id, entity_db in entity_keys:
            if entity_db not in entities_by_db:
                entities_by_db[entity_db] = []
            entities_by_db[entity_db].append(entity_id)

        # Fetch data for each database separately
        for database, entity_ids in entities_by_db.items():
            if not entity_ids:
                continue

            # Batch fetch entity data using XML API
            # IDs must be from the same database for XML API
            entries_across_dbs.update(self._client.fetch_entries(entity_ids))

        return entries_across_dbs

    def _populate_hierarchy_fields(
        self,
        hierarchies: List[Dict[str, Any]],
        entity_data_map: Dict[str, Dict[str, Any]],
    ) -> None:
        """Populate fields in hierarchy nodes from fetched entity data.

        Args:
            hierarchies: List of hierarchy dictionaries (modified in place)
            entity_data_map: Dictionary mapping entity_id to entity data
        """
        def populate_node(node: Dict[str, Any]) -> None:
            """Recursively populate fields in a node."""
            node["fields"] = entity_data_map[node.get("id")]

            # Recursively populate cross-references
            for child in node.get("crossref", []):
                populate_node(child)

        for hierarchy in hierarchies:
            populate_node(hierarchy)
