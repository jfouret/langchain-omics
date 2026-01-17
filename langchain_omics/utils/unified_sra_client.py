"""Unified SRA client using EBI Search and ENA Portal APIs."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from typing_extensions import Annotated
from urllib.parse import urlencode

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field, validate_call

from .ena_xml_parser import parse_ena_xml

class _UndirectedGraph:
    def __init__(self) -> None:
        self._adj: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    
    def add_edge(self, n1: Tuple[str, str], n2: Tuple[str, str]) -> None:
        if n1 not in self._adj:
            self._adj[n1] = set()
        if n2 not in self._adj:
            self._adj[n2] = set()
        self._adj[n1].add(n2)
        self._adj[n2].add(n1)
    
    def has_edge(self, n1: Tuple[str, str], n2: Tuple[str, str]) -> bool:
        return n2 in self._adj.get(n1, set())
    
    def get_neighbors(self, node: Tuple[str, str]) -> List[Tuple[str, str]]:
        return list(self._adj.get(node, []))
    
    def get_linked_nodes(self) -> List[Tuple[str, str]]:
        return list(self._adj.keys())

class UnifiedSRAClientConfig(BaseModel):
    """Private configuration class for UnifiedSRAClient with field validation."""
    
    request_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout in seconds for HTTP requests"
    )
    rate_limit_delay: float = Field(
        default=0,
        ge=0,
        le=60,
        description="Delay in seconds between requests for rate limiting"
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        le=100,
        description="Maximum number of entries per batch request"
    )
    search_size: int = Field(
        default=100,
        ge=1,
        le=100,
        description="Number of results to fetch per page in search/crossref operations"
    )
    default_max_crossref: int = Field(
        default=1000000,
        ge=1,
        le=1000000,
        description="Default maximum number of cross-references to return per entry"
    )
    default_max_search_hits: int = Field(
        default=100,
        ge=1,
        le=1000000,
        description="Default maximum number of search results to return"
    )
    log_level: str = Field(
        default="WARNING",
        pattern="^(DEBUG|INFO|WARNING|ERROR)$",
        description="Logging level - DEBUG, INFO, WARNING, or ERROR"
    )

class UnifiedSRAClient:
    """Unified SRA client using EBI Search and ENA Browser APIs.

    This client combines the strengths of both APIs:
    - EBI Search: Simple queries, efficient cross-references
    - ENA Browser: Full XML data retrieval with rich nested structures

    Attributes:
        _config: Private configuration object with validated parameters
    """

    _ebi_search_base_url: str = "https://www.ebi.ac.uk/ebisearch/ws/rest"
    _ena_portal_base_url: str = "https://www.ebi.ac.uk/ena/portal/api"
    _ena_browser_base_url: str = "https://www.ebi.ac.uk/ena/browser/api"

    _headers: Dict[str, str] = {
        "User-Agent": "langchain-omics"
    }

    _relationships: Dict[str, Dict[str, List[str]]] = {
        "sra-study": ["sra-experiment"],
        "sra-experiment": ["sra-study", "sra-sample", "sra-run"],
        "sra-sample": ["sra-experiment"],
        "sra-run": ["sra-experiment"]
    }

    _valid_domains = {"sra-study", "sra-experiment", "sra-sample", "sra-run"}

    def __init__(
        self,
        config: Optional[UnifiedSRAClientConfig] = None
    ):
        """Initialize UnifiedSRAClient with validated configuration."""
        self._config = config or UnifiedSRAClientConfig()
        
        # Setup logging
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(getattr(logging, self._config.log_level))
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)


    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None, mimetype: str = "json") -> Any:
        """Make HTTP request with error handling and rate limiting.
        
        Args:
            url: Request URL
            params: Optional query parameters
            mimetype: Response type - 'xml' or 'json' (default: 'json')
        
        Returns:
            JSON response or raw XML text if mimetype='xml'
        """
        if params:
            clean_params = {k: str(v) for k, v in params.items() if v is not None}
            if clean_params:
                url = f"{url}?{urlencode(clean_params)}"

        self._logger.log(5, f"TRACE: Making request to {url} with mimetype={mimetype}")

        time.sleep(self._config.rate_limit_delay)

        # Set appropriate headers based on mimetype
        headers = dict(self._headers)
        if mimetype == "xml":
            headers["Accept"] = "application/xml"
        elif mimetype == "json":
            headers["Accept"] = "application/json"
        else:
            raise ValueError(f"Invalid mimetype: {mimetype}. Must be 'xml' or 'json'")

        try:
            response = requests.get(
                url, headers=headers, timeout=self._config.request_timeout
            )

            # Handle 204 No Content (end of pagination)
            if response.status_code == 204:
                return [] if mimetype == "json" else ""

            response.raise_for_status()
            
            if mimetype == "xml":
                return response.text
            else:
                return response.json()

        except requests.exceptions.RequestException as e:
            raise requests.RequestException(f"API request failed: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid JSON response: {e}")

    def _fetch_entries_batch(
        self, entry_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Use ENA Browser XML API for entry retrieval (private batch method).

        Args:
            entry_ids: List of entry IDs (max 100 per batch)

        Returns:
            Dict of entry dictionaries with full XML-parsed data, keyed by accession
        """
        if not entry_ids:
            return dict()

        # Limit to 100 entries per batch (ENA Browser limit)
        assert len(entry_ids) <= self._config.batch_size

        self._logger.debug(
            f"Getting entries batch: {len(entry_ids)} entries"
        )

        # Build URL for ENA Browser XML API
        url = f"{self._ena_browser_base_url}/xml/{','.join(entry_ids)}"
        params = {
            "download": "false",
            "gzip": "true",
            "includeLinks": "false"
        }

        # Fetch XML data from ENA Browser API
        try:
            xml_response = self._make_request(url, params, mimetype="xml")
            self._logger.debug(f"Retrieved XML response for {len(entry_ids)} entries")
        except Exception as e:
            self._logger.error(f"Get entries XML failed: {e}")
            raise

        if not xml_response:
            self._logger.debug("Empty XML response")
            if len(entry_ids) > 0:
                exit(1)
            return dict()

        # Parse XML using the XML parser
        try:
            parsed_dict = parse_ena_xml(xml_response)
            self._logger.debug(f"Parsed XML into {len(parsed_dict)} entries")
            return parsed_dict
        except Exception as e:
            self._logger.error(f"Failed to parse XML response: {e}")
            return dict()

    @validate_call
    def fetch_entries(
        self,
        entry_ids: Annotated[
            List[str],
            Field(min_length=1, description="List of entry IDs to fetch")
        ]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch entries in batches of 100 (ENA Browser limit).

        Args:
            entry_ids: List of entry IDs (any size)

        Returns:
            Dict of all entry dictionaries, keyed by accession
        """
        if not entry_ids:
            return dict()

        self._logger.debug(
            f"Getting entries: {len(entry_ids)} total entries"
        )
        all_entries = dict()

        # Process in batches of self.batch_size
        for i in range(0, len(entry_ids), self._config.batch_size):
            batch_ids = entry_ids[i : i + self._config.batch_size]
            self._logger.debug(
                f"Processing batch from {i}: {len(batch_ids)} entries"
            )
            batch_entries = self._fetch_entries_batch(batch_ids)
            all_entries.update(batch_entries)

        self._logger.debug(f"Retrieved {len(all_entries)} total entries")
        return all_entries

    def _fetch_crossref_batch(
        self,
        domain: str,
        entry_ids: List[str],
        target_domain: str,
        max_size: Optional[int] = None
    ) -> Dict[str, Set[str]]:
        """Use EBI Search for cross-references with pagination support.

        Args:
            domain: Source domain (e.g., 'sra-sample')
            entry_ids: List of source entry IDs (max self._config.batch_size)
            target_domain: Target domain (e.g., 'sra-experiment')
            max_size: Maximum number of cross-references to return per entry (default: default_max_crossref)

        Returns:
            Dictionary mapping entry_id -> list of cross-referenced IDs
        """
        if not entry_ids:
            return {}

        # Limit to 100 entries per batch
        assert len(entry_ids) <= self._config.batch_size

        if max_size is None:
            max_size = self._config.default_max_crossref

        all_results = {}
        entries_to_process = list(entry_ids)
        start: int = 0

        # 
        while len(entries_to_process) > 0 and start < max_size:

            url = (
                f"{self._ebi_search_base_url}/{domain}/entry/"
                f"{','.join(entries_to_process)}/xref/{target_domain}"
            )
            
            entries_to_process = []
            start = start + self._config.search_size

            if start + self._config.search_size > max_size:
                search_size = max_size - start
            else:
                search_size = self._config.search_size

            # Use search_size for the API request (page size)
            params = {"format": "json", "size": str(search_size), "start": str(start)}

            self._logger.debug(
                f"Cross-reference batch call: {domain} -> {target_domain}, "
                f"{len(entry_ids)} entries, max_size={max_size}"
            )
        
            try:
                response = self._make_request(url, params)
                entries = response.get("entries", [])

                # Build mapping of source_id -> [target_ids] and check for pagination
                for entry in entries:
                    # The source_id is actually the acc/id of the entry we queried
                    s_id = entry.get("acc") or entry.get("id")
                    if s_id:
                        cross_refs = set()
                        references = entry.get("references", [])
                        for ref in references:
                            acc = ref.get("acc")
                            if acc:
                                cross_refs.add(acc)
                        if s_id not in all_results.keys():
                            all_results[s_id] = cross_refs
                        else:
                            all_results[s_id] = all_results[s_id] | cross_refs

                        reference_count = entry.get("referenceCount", 0)
                        # start is the next 0-based start
                        if reference_count > start:
                            entries_to_process.append(s_id)

            except Exception as e:
                self._logger.error(f"Cross-reference batch chunk failed: {e}")
                # Add empty results for failed chunk
                raise

        # add empty if not returned
        for entry in entry_ids:
            if entry not in all_results.keys():
                all_results[entry] = []

        return all_results

    @validate_call
    def fetch_crossref(
        self,
        domain: Annotated[
            str,
            Field(
                pattern=r"^sra-(study|experiment|sample|run)$",
                description="SRA domain name"
            )
        ],
        entry_ids: Annotated[
            List[str],
            Field(
                min_length=1,
                description="List of source entry IDs"
            )
        ],
        target_domain: Annotated[
            str,
            Field(
                pattern=r"^sra-(study|experiment|sample|run)$",
                description="Target SRA domain name"
            )
        ],
        max_size: Annotated[
            Optional[int],
            Field(
                ge=1,
                le=1000000,
                description="Maximum cross-references per entry"
            )
        ] = None
    ) -> Dict[str, Set[str]]:
        """Use EBI Search for cross-references with pagination support.

        Args:
            domain: Source domain (e.g., 'sra-sample')
            entry_ids: List of source entry IDs (any size)
            target_domain: Target domain (e.g., 'sra-experiment')
            max_size: Maximum number of cross-references to return per entry (default: default_max_crossref)

        Returns:
            Dictionary mapping entry_id -> list of cross-referenced IDs
        """
        if not entry_ids:
            return {}

        all_results = {}

        # Process in chunks of self._config.batch_size
        for i in range(0, len(entry_ids), self._config.batch_size):
            batch_ids = entry_ids[i : i + self._config.batch_size]
            batch_results = self._fetch_crossref_batch(
                domain, batch_ids, target_domain, max_size
            )
            all_results.update(batch_results)

        return all_results

    @validate_call
    def search_domain(
        self,
        domain: Annotated[
            str,
            Field(
                pattern=r"^sra-(study|experiment|sample|run)$",
                description="SRA domain name"
            )
        ],
        query: Annotated[
            str,
            Field(
                min_length=1,
                description="Search query string"
            )
        ],
        max_size: Annotated[
            Optional[int],
            Field(
                ge=1,
                le=1000000,
                description="Maximum number of results to return"
            )
        ] = None
    ) -> List[Dict[str, Any]]:
        """Use EBI Search for simple domain search with pagination support.

        Args:
            domain: SRA domain (e.g., 'sra-study', 'sra-sample')
            query: Search query string
            max_size: Maximum number of results to return (default: default_max_search_hits)

        Returns:
            List of entry dictionaries
        """
        if max_size is None:
            max_size = self._config.default_max_search_hits

        self._logger.debug(
            f"Searching domain {domain} with query '{query}', max_size={max_size}"
        )

        all_entries = []
        start = 0

        while len(all_entries) < max_size:
            # Determine how many results to fetch in this request
            fetch_size = min(self._config.search_size, max_size - len(all_entries))

            url = f"{self._ebi_search_base_url}/{domain}"
            params = {
                "query": query,
                "size": str(fetch_size),
                "start": str(start),
                "format": "json"
            }

            try:
                response = self._make_request(url, params)
                entries = response.get("entries", [])
                hit_count = response.get("hitCount", 0)

                if not entries:
                    # No more results available
                    break

                all_entries.extend(entries)
                self._logger.debug(
                    f"Fetched {len(entries)} entries (total: {len(all_entries)}/{max_size}, "
                    f"hitCount: {hit_count})"
                )

                # Check if we've fetched all available results
                if start + len(entries) >= hit_count:
                    break

                start += len(entries)

            except Exception as e:
                self._logger.debug(f"Domain search failed for {domain}: {e}")
                break

        self._logger.debug(
            f"Domain search returned {len(all_entries)} entries for {domain}"
        )
        return all_entries

    @validate_call
    def search_multiple_domains(
        self,
        databases: Annotated[
            List[str],
            Field(
                min_length=1,
                description="List of database names to search"
            )
        ],
        query: Annotated[
            str,
            Field(
                min_length=1,
                description="Search query string"
            )
        ],
        max_size: Annotated[
            Optional[int],
            Field(
                ge=1,
                le=1000000,
                description="Default maximum results per database"
            )
        ] = None,
        max_size_per_db: Annotated[
            Optional[Dict[str, int]],
            Field(description="Custom max_size per database")
        ] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple databases and return results by database.

        Args:
            databases: List of database names to search
                (e.g., ["sra-study", "sra-sample"])
            query: Search query string
            max_size: Default maximum number of results per database
                (used if not in max_size_per_db, default: default_max_search_hits)
            max_size_per_db: Optional dictionary mapping database name to custom max_size limit
                        If database is not in keys, uses the default max_size parameter

        Returns:
            Dictionary mapping database name to list of entry dictionaries.
            Maintains search order by processing databases in the order provided.
        """
        self._logger.debug(f"Searching multiple databases: {databases}")
        results = {}

        for database in databases:
            assert database in self._valid_domains
            
            # Determine max_size for this database
            if max_size_per_db:
                db_max_size = max_size_per_db.get(database, max_size)
            else:
                db_max_size = max_size

            self._logger.debug(f"Searching database: {database} with max_size={db_max_size}")
            entries = self.search_domain(database, query, max_size=db_max_size)
            results[database] = entries
            self._logger.debug(f"Found {len(entries)} entries in {database}")

        return results

    @validate_call
    def fetch_crossref_recursively(
        self,
        entities: Annotated[
            List[Tuple[str, str]],
            Field(
                min_length=1,
                description="List of (ID, DATABASE) tuples to start from"
            )
        ],
        max_depth: Annotated[
            int,
            Field(
                ge=1,
                le=10,
                description="Maximum recursion depth"
            )
        ] = 2,
        max_size: Annotated[
            Optional[int],
            Field(
                ge=1,
                le=1000000,
                description="Maximum cross-references per entry"
            )
        ] = None
    ) -> _UndirectedGraph:
        """Recursively build cross-reference relations with depth limit.

        Args:
            entities: List of (ID, DATABASE) tuples to start from
            max_depth: Maximum recursion depth (default: 2)
            max_size: Maximum number of cross-references to fetch per entry 
            (default: default_max_crossref)

        Returns:
            
        """

        graph = _UndirectedGraph()

        def _recursive_walk(
            _graph: _UndirectedGraph,
            _entities: Dict[str, Set[str]],
            _fetched: Dict[str, Set[str]],
            _depth: int,
        ) -> _UndirectedGraph:
            """Recursive helper function to build the graph."""

            # quit recursion if max depth was reached
            if _depth >= max_depth:
                return _graph
            
            # prepare next recursion
            next_entities = {}
            next_count: int = 0
            for domain in self._valid_domains:
                next_entities[domain] = set()

            # iterate over all domains
            for domain, queries in _entities.items():

                # iterate over possible target domain for each domain
                for target_domain in self._relationships[domain]:

                    crossref = self.fetch_crossref(
                        domain, list(queries), target_domain, max_size)
                    
                    for source_id, target_ids in crossref.items():
                        # feed next entities to fetch
                        next_count += len(target_ids)
                        new_entities = target_ids - _fetched[target_domain]
                        next_entities[target_domain] = \
                            next_entities[target_domain] | new_entities
                        
                        # add edges to graph
                        for target_id in target_ids:
                            _graph.add_edge(
                                (source_id, domain),
                                (target_id, target_domain)
                            )
            # next recursion only if 
            if next_count > 0:
                _graph = _recursive_walk(_graph, next_entities, _fetched, _depth + 1)
            return _graph
        init_fetched = {}
        for domain in self._valid_domains:
            init_fetched[domain] = set()
        init_queries = dict(init_fetched)
        for entity in entities:
            init_queries[entity[1]].add(entity[0])
        graph = _recursive_walk(graph, init_queries, init_fetched, 0)
        return graph
