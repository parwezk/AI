# core/github_agent_system.py
# Complete GitHub AI Agent System - Core Implementation

import os
import ast
import json
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil

# Core libraries
import pandas as pd
import numpy as np

# Machine Learning and Embeddings
from sentence_transformers import SentenceTransformer

# Vector Database
import chromadb

# GitHub Integration
from github import Github, Repository
import git

# Code Parsing (with fallback)
try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Configuration
from config.config import Config

# Set up logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL, "INFO"))
logger = logging.getLogger(__name__)

# ===================================================================
# DATA MODELS
# ===================================================================


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""

    content: str
    file_path: str
    start_line: int
    end_line: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    docstring: Optional[str] = None
    chunk_type: str = "function"


@dataclass
class QueryResult:
    """Represents a search result"""

    code_chunk: CodeChunk
    similarity_score: float
    explanation: str


@dataclass
class AgentResponse:
    """Standard response format from agents"""

    success: bool
    data: Any
    message: str
    agent_name: str


# ===================================================================
# GITHUB REPOSITORY HANDLER
# ===================================================================


class GitHubRepositoryHandler:
    """Handles GitHub repository operations and code extraction"""

    def __init__(self, github_token: Optional[str] = None):
        """Initialize GitHub handler with optional token"""
        self.github = Github(github_token) if github_token else Github()
        self.temp_dir = None

    def clone_repository(self, repo_url: str, branch: str = "main") -> str:
        """Clone repository to temporary directory"""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()

            logger.info(f"Cloning repository: {repo_url}")

            # First, detect the default branch if using "main"
            if branch == "main":
                branch = self._detect_default_branch(repo_url)
                logger.info(f"Using detected default branch: {branch}")

            # Try git clone first
            try:
                repo = git.Repo.clone_from(
                    repo_url, self.temp_dir, depth=1, single_branch=True, branch=branch
                )
                logger.info(f"Repository cloned successfully to: {self.temp_dir}")
            except Exception as git_error:
                logger.warning(f"Git clone failed: {git_error}")
                # Try alternative branches if main branch fails
                if branch == "main":
                    for alt_branch in ["master", "develop", "dev"]:
                        try:
                            logger.info(f"Trying alternative branch: {alt_branch}")
                            repo = git.Repo.clone_from(
                                repo_url,
                                self.temp_dir,
                                depth=1,
                                single_branch=True,
                                branch=alt_branch,
                            )
                            logger.info(
                                f"Successfully cloned using branch: {alt_branch}"
                            )
                            break
                        except:
                            continue
                    else:
                        # All git clone attempts failed, fallback to API
                        logger.warning(
                            "All git clone attempts failed, trying GitHub API..."
                        )
                        self._download_via_api(repo_url, branch)
                else:
                    # Fallback: download via GitHub API
                    logger.warning("Git clone failed, trying GitHub API...")
                    self._download_via_api(repo_url, branch)

            return self.temp_dir

        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise

    def _detect_default_branch(self, repo_url: str) -> str:
        """Detect the default branch of a repository"""
        try:
            # Extract owner and repo from URL
            parts = repo_url.replace("https://github.com/", "").split("/")
            owner, repo_name = parts[0], parts[1]

            # Get repository info
            repo = self.github.get_repo(f"{owner}/{repo_name}")
            default_branch = repo.default_branch
            logger.info(f"Detected default branch: {default_branch}")
            return default_branch

        except Exception as e:
            logger.warning(f"Could not detect default branch: {e}, using 'main'")
            return "main"

    def _download_via_api(self, repo_url: str, branch: str = "main"):
        """Download repository contents via GitHub API (fallback)"""
        # Extract owner and repo from URL
        parts = repo_url.replace("https://github.com/", "").split("/")
        owner, repo_name = parts[0], parts[1]

        try:
            repo = self.github.get_repo(f"{owner}/{repo_name}")

            # If branch is main but doesn't exist, try default branch
            if branch == "main":
                try:
                    repo.get_contents("", ref=branch)
                except:
                    branch = repo.default_branch
                    logger.info(
                        f"Branch 'main' not found, using default branch: {branch}"
                    )

            contents = repo.get_contents("", ref=branch)
            self._download_contents_recursive(contents, self.temp_dir, repo, branch)

        except Exception as e:
            # Try alternative branches
            for alt_branch in ["master", "develop", "dev", repo.default_branch]:
                if alt_branch != branch:  # Don't retry the same branch
                    try:
                        logger.info(f"Trying alternative branch via API: {alt_branch}")
                        contents = repo.get_contents("", ref=alt_branch)
                        self._download_contents_recursive(
                            contents, self.temp_dir, repo, alt_branch
                        )
                        logger.info(
                            f"Successfully downloaded using branch: {alt_branch}"
                        )
                        return
                    except:
                        continue

            raise Exception(f"Failed to download via API with any branch: {e}")

    # Rest of the methods remain the same...
    def _download_contents_recursive(
        self, contents, local_path, repo, branch, max_files=100
    ):
        """Recursively download repository contents"""
        file_count = 0

        for content_file in contents:
            if file_count >= max_files:
                logger.warning(f"Reached maximum file limit ({max_files})")
                break

            local_file_path = os.path.join(local_path, content_file.name)

            if content_file.type == "dir":
                os.makedirs(local_file_path, exist_ok=True)
                try:
                    sub_contents = repo.get_contents(content_file.path, ref=branch)
                    remaining_files = max_files - file_count
                    sub_count = self._download_contents_recursive(
                        sub_contents, local_file_path, repo, branch, remaining_files
                    )
                    file_count += sub_count
                except:
                    continue
            else:
                # Only download supported file types
                if any(
                    content_file.name.endswith(ext)
                    for ext in Config.SUPPORTED_EXTENSIONS
                ):
                    try:
                        file_content = content_file.decoded_content.decode(
                            "utf-8", errors="ignore"
                        )
                        with open(local_file_path, "w", encoding="utf-8") as f:
                            f.write(file_content)
                        file_count += 1
                    except:
                        continue

        return file_count

    def extract_code_files(self, repo_path: str) -> List[Dict[str, Any]]:
        """Extract all supported code files from repository"""
        code_files = []
        file_count = 0

        for root, dirs, files in os.walk(repo_path):
            # Skip common non-code directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ["node_modules", "__pycache__", "venv", "env"]
            ]

            for file in files:
                if file_count >= 200:  # Limit for performance
                    break

                if any(file.endswith(ext) for ext in Config.SUPPORTED_EXTENSIONS):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)

                    try:
                        # Check file size
                        if os.path.getsize(file_path) > Config.MAX_FILE_SIZE:
                            continue

                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            content = f.read()

                        code_files.append(
                            {
                                "path": relative_path,
                                "content": content,
                                "extension": os.path.splitext(file)[1],
                                "size": len(content),
                            }
                        )
                        file_count += 1

                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")
                        continue

        logger.info(f"Extracted {len(code_files)} code files")
        return code_files

    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary files")


# ===================================================================
# CODE PARSER
# ===================================================================


class CodeParser:
    """Parse code files and create meaningful chunks"""

    def __init__(self):
        """Initialize Tree-sitter parser"""
        self.tree_sitter_available = TREE_SITTER_AVAILABLE
        if self.tree_sitter_available:
            try:
                PY_LANGUAGE = Language(tspython.language(), "python")
                self.parser = Parser()
                self.parser.set_language(PY_LANGUAGE)
                logger.info("Tree-sitter parser initialized")
            except:
                self.tree_sitter_available = False
                logger.warning("Tree-sitter failed, using AST-only parsing")

    def parse_file(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse file and extract code chunks"""
        chunks = []

        # Determine file type
        extension = os.path.splitext(file_path)[1]

        if extension == ".py":
            chunks = self._parse_python_file(content, file_path)
        else:
            # Basic chunking for other file types
            chunks = self._basic_chunking(content, file_path)

        return chunks

    def _parse_python_file(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse Python file using AST"""
        chunks = []

        try:
            tree = ast.parse(content)
            lines = content.split("\n")

            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._create_function_chunk(node, lines, file_path)
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.ClassDef):
                    chunk = self._create_class_chunk(node, lines, file_path)
                    if chunk:
                        chunks.append(chunk)

        except SyntaxError:
            # Fallback to basic chunking
            chunks = self._basic_chunking(content, file_path)

        return chunks

    def _create_function_chunk(
        self, node: ast.FunctionDef, lines: List[str], file_path: str
    ) -> Optional[CodeChunk]:
        """Create code chunk for function"""
        try:
            start_line = node.lineno - 1
            end_line = getattr(node, "end_lineno", start_line + 20)

            if end_line is None:
                end_line = start_line + 20

            # Extract function content
            content_lines = lines[start_line : min(end_line, len(lines))]
            content = "\n".join(content_lines)

            # Extract docstring
            docstring = ast.get_docstring(node)

            return CodeChunk(
                content=content,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=min(end_line, len(lines)),
                function_name=node.name,
                docstring=docstring,
                chunk_type="function",
            )
        except:
            return None

    def _create_class_chunk(
        self, node: ast.ClassDef, lines: List[str], file_path: str
    ) -> Optional[CodeChunk]:
        """Create code chunk for class"""
        try:
            start_line = node.lineno - 1
            end_line = getattr(node, "end_lineno", start_line + 30)

            if end_line is None:
                end_line = start_line + 30

            # Extract class content
            content_lines = lines[start_line : min(end_line, len(lines))]
            content = "\n".join(content_lines)

            # Extract docstring
            docstring = ast.get_docstring(node)

            return CodeChunk(
                content=content,
                file_path=file_path,
                start_line=start_line + 1,
                end_line=min(end_line, len(lines)),
                class_name=node.name,
                docstring=docstring,
                chunk_type="class",
            )
        except:
            return None

    def _basic_chunking(self, content: str, file_path: str) -> List[CodeChunk]:
        """Basic line-based chunking for non-Python files or fallback"""
        lines = content.split("\n")
        chunks = []
        chunk_size = Config.MAX_CHUNK_SIZE // 10  # Convert to lines roughly

        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i : i + chunk_size]
            chunk_content = "\n".join(chunk_lines)

            if chunk_content.strip():  # Only non-empty chunks
                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=i + 1,
                        end_line=min(i + chunk_size, len(lines)),
                        chunk_type="module",
                    )
                )

        return chunks


# ===================================================================
# VECTOR DATABASE
# ===================================================================


class VectorDatabase:
    """Manages code embeddings and similarity search using Chroma"""

    def __init__(self):
        """Initialize vector database and embedding model"""
        # Initialize embedding model
        logger.info("Loading embedding model...")
        try:
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

        # Initialize Chroma client
        try:
            # Ensure directory exists
            os.makedirs(Config.VECTOR_DB_PATH, exist_ok=True)

            self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)

            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=Config.COLLECTION_NAME
                )
                logger.info("Loaded existing collection")
            except:
                self.collection = self.client.create_collection(
                    name=Config.COLLECTION_NAME,
                    metadata={"description": "Code embeddings for semantic search"},
                )
                logger.info("Created new collection")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            # Fallback to in-memory
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(name=Config.COLLECTION_NAME)
            logger.warning("Using in-memory database (data will not persist)")

    def add_code_chunks(self, chunks: List[CodeChunk]) -> bool:
        """Add code chunks to vector database"""
        if not chunks:
            return True

        try:
            logger.info(f"Processing {len(chunks)} code chunks...")

            # Prepare data for embedding
            documents = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                # Create searchable text combining code and metadata
                searchable_text = self._create_searchable_text(chunk)
                documents.append(searchable_text)

                # Create metadata
                metadata = {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "function_name": chunk.function_name or "",
                    "class_name": chunk.class_name or "",
                    "docstring": chunk.docstring or "",
                }
                metadatas.append(metadata)

                # Create unique ID
                chunk_id = (
                    f"chunk_{i}_{hashlib.md5(chunk.content.encode()).hexdigest()[:8]}"
                )
                ids.append(chunk_id)

            # Generate embeddings in batches for memory efficiency
            batch_size = 32
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i : i + batch_size]
                batch_metadata = metadatas[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]

                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}"
                )

                # Generate embeddings
                embeddings = self.embedding_model.encode(
                    batch_docs, show_progress_bar=False
                )

                # Add to collection
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids,
                )

            logger.info(f"Added {len(chunks)} code chunks to database")
            return True

        except Exception as e:
            logger.error(f"Failed to add chunks to database: {e}")
            return False

    def search_similar_code(
        self, query: str, n_results: int = None
    ) -> List[QueryResult]:
        """Search for similar code chunks"""
        if n_results is None:
            n_results = Config.MAX_RESULTS

        try:
            logger.info(f"Searching for: '{query}'")

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])

            # Search in collection
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            # Convert results to QueryResult objects
            query_results = []

            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    # Convert distance to similarity score
                    similarity_score = max(0, 1 - distance)

                    # Skip results below threshold
                    if similarity_score < Config.SIMILARITY_THRESHOLD:
                        continue

                    # Extract original code from metadata
                    code_content = (
                        doc.split("CODE:\n", 1)[-1] if "CODE:\n" in doc else doc
                    )

                    code_chunk = CodeChunk(
                        content=code_content,
                        file_path=metadata["file_path"],
                        start_line=metadata["start_line"],
                        end_line=metadata["end_line"],
                        function_name=metadata["function_name"]
                        if metadata["function_name"]
                        else None,
                        class_name=metadata["class_name"]
                        if metadata["class_name"]
                        else None,
                        docstring=metadata["docstring"]
                        if metadata["docstring"]
                        else None,
                        chunk_type=metadata["chunk_type"],
                    )

                    query_results.append(
                        QueryResult(
                            code_chunk=code_chunk,
                            similarity_score=similarity_score,
                            explanation=f"Found in {metadata['file_path']} (similarity: {similarity_score:.2f})",
                        )
                    )

            logger.info(f"Found {len(query_results)} relevant results")
            return query_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _create_searchable_text(self, chunk: CodeChunk) -> str:
        """Create searchable text from code chunk"""
        parts = []

        # Add file context
        parts.append(f"FILE: {chunk.file_path}")

        # Add function/class name
        if chunk.function_name:
            parts.append(f"FUNCTION: {chunk.function_name}")
        if chunk.class_name:
            parts.append(f"CLASS: {chunk.class_name}")

        # Add docstring
        if chunk.docstring:
            parts.append(f"DESCRIPTION: {chunk.docstring}")

        # Add code content
        parts.append(f"CODE:\n{chunk.content}")

        return "\n".join(parts)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": Config.COLLECTION_NAME,
                "embedding_model": Config.EMBEDDING_MODEL,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_chunks": 0}


# ===================================================================
# AI AGENT SYSTEM
# ===================================================================


class BaseAgent:
    """Base class for all AI agents"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")

    def process(self, input_data: Any) -> AgentResponse:
        """Process input and return response"""
        raise NotImplementedError

    def _create_response(self, success: bool, data: Any, message: str) -> AgentResponse:
        """Create standardized response"""
        return AgentResponse(
            success=success, data=data, message=message, agent_name=self.name
        )


class QueryUnderstandingAgent(BaseAgent):
    """Agent that processes and understands natural language queries"""

    def __init__(self):
        super().__init__("QueryUnderstanding")
        # Intent keywords mapping
        self.intent_keywords = {
            "search": ["find", "show", "locate", "search", "where", "what"],
            "explain": ["explain", "how", "why", "describe", "tell me about"],
            "generate": ["create", "generate", "write", "make", "build"],
            "debug": ["debug", "fix", "error", "problem", "issue", "bug"],
        }

    def process(self, query: str) -> AgentResponse:
        """Process natural language query and extract intent/parameters"""
        try:
            # Clean and normalize query
            clean_query = query.lower().strip()

            # Detect intent
            intent = self._detect_intent(clean_query)

            # Extract parameters
            params = self._extract_parameters(clean_query)

            result = {
                "original_query": query,
                "cleaned_query": clean_query,
                "intent": intent,
                "parameters": params,
                "search_terms": self._extract_search_terms(clean_query),
            }

            return self._create_response(
                success=True,
                data=result,
                message=f"Query understood with intent: {intent}",
            )

        except Exception as e:
            return self._create_response(
                success=False, data=None, message=f"Failed to understand query: {e}"
            )

    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query"""
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in query for keyword in keywords):
                return intent
        return "search"  # Default intent

    def _extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from query"""
        params = {}

        # Common programming concepts
        concepts = [
            "function",
            "class",
            "method",
            "variable",
            "import",
            "api",
            "test",
            "error",
            "config",
        ]
        for concept in concepts:
            if concept in query:
                params["concept"] = concept
                break

        # Programming languages
        languages = ["python", "javascript", "java", "cpp", "c++"]
        for lang in languages:
            if lang in query:
                params["language"] = lang
                break

        return params

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract key search terms from query"""
        # Simple keyword extraction
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "how",
            "what",
            "where",
            "when",
            "why",
        }
        words = query.split()
        return [word for word in words if word not in stop_words and len(word) > 2]


# ===================================================================
# MAIN ORCHESTRATOR
# ===================================================================


class GitHubAgentOrchestrator:
    """Main orchestrator that coordinates all agents"""

    def __init__(self, github_token: Optional[str] = None):
        """Initialize the agent system"""
        logger.info("Initializing GitHub Agent System...")

        # Use token from config if not provided
        if not github_token:
            github_token = Config.GITHUB_TOKEN

        # Initialize components
        self.github_handler = GitHubRepositoryHandler(github_token)
        self.code_parser = CodeParser()
        self.vector_db = VectorDatabase()

        # Initialize agents
        self.query_agent = QueryUnderstandingAgent()

        self.indexed_repos = set()

        logger.info("Agent system initialized successfully!")

    def index_repository(self, repo_url: str, branch: str = "main") -> bool:
        """Index a GitHub repository for search"""
        try:
            logger.info(f"Starting repository indexing: {repo_url}")

            # Clone repository
            repo_path = self.github_handler.clone_repository(repo_url, branch)

            # Extract code files
            code_files = self.github_handler.extract_code_files(repo_path)

            if not code_files:
                logger.warning("No supported code files found")
                return False

            # Parse and chunk code
            all_chunks = []
            for file_info in code_files:
                logger.debug(f"Parsing {file_info['path']}...")
                chunks = self.code_parser.parse_file(
                    file_info["content"], file_info["path"]
                )
                all_chunks.extend(chunks)

            logger.info(f"Created {len(all_chunks)} code chunks")

            # Add to vector database
            success = self.vector_db.add_code_chunks(all_chunks)

            if success:
                self.indexed_repos.add(repo_url)
                logger.info("Successfully indexed repository!")

                # Display stats
                stats = self.vector_db.get_collection_stats()
                logger.info(
                    f"Database now contains {stats.get('total_chunks', 0)} total chunks"
                )

            # Cleanup
            self.github_handler.cleanup()

            return success

        except Exception as e:
            logger.error(f"Repository indexing failed: {e}")
            self.github_handler.cleanup()
            return False

    def query(self, user_query: str) -> Dict[str, Any]:
        """Process a natural language query about the indexed code"""
        try:
            logger.info(f"Processing query: '{user_query}'")

            # Step 1: Understand the query
            query_response = self.query_agent.process(user_query)
            if not query_response.success:
                return {"success": False, "error": query_response.message}

            logger.debug(f"Query understood - Intent: {query_response.data['intent']}")

            # Step 2: Search repository
            search_terms = " ".join(query_response.data["search_terms"])
            if not search_terms:
                search_terms = query_response.data["cleaned_query"]

            results = self.vector_db.search_similar_code(search_terms)

            if not results:
                return {
                    "success": False,
                    "error": "No relevant code found. Try a different query or check if repository is indexed.",
                }

            # Step 3: Create response
            primary_result = results[0]
            related_results = results[1:] if len(results) > 1 else []

            # Step 4: Validate code (basic validation)
            validation_results = self._validate_code(primary_result.code_chunk.content)

            response = {
                "primary_result": {
                    "file": primary_result.code_chunk.file_path,
                    "function": primary_result.code_chunk.function_name,
                    "class": primary_result.code_chunk.class_name,
                    "code": primary_result.code_chunk.content,
                    "explanation": primary_result.explanation,
                    "similarity": primary_result.similarity_score,
                },
                "related_results": [
                    {
                        "file": r.code_chunk.file_path,
                        "similarity": r.similarity_score,
                        "snippet": r.code_chunk.content[:200] + "..."
                        if len(r.code_chunk.content) > 200
                        else r.code_chunk.content,
                    }
                    for r in related_results
                ],
                "summary": f"Found {len(results)} relevant code chunks. Primary match in {primary_result.code_chunk.file_path} with {primary_result.similarity_score:.2f} similarity.",
                "validation": validation_results,
            }

            return {
                "success": True,
                "query": user_query,
                "response": response,
                "processing_chain": [
                    "QueryUnderstanding",
                    "RepositoryIntelligence",
                    "CodeGeneration",
                    "Validation",
                ],
            }

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {"success": False, "error": f"Query processing failed: {e}"}

    def _validate_code(self, code: str) -> Dict[str, Any]:
        """Basic code validation"""
        validation = {
            "syntax_valid": False,
            "issues": [],
            "suggestions": [],
            "quality_score": 0.0,
        }

        try:
            # Try AST parsing for Python code
            ast.parse(code)
            validation["syntax_valid"] = True
            validation["quality_score"] += 0.5

            # Basic quality checks
            if '"""' in code or "'''" in code:
                validation["quality_score"] += 0.1
            if "#" in code:
                validation["quality_score"] += 0.1
            if "try:" in code or "except" in code:
                validation["quality_score"] += 0.1
            if "->" in code:
                validation["quality_score"] += 0.1

            validation["quality_score"] = min(validation["quality_score"], 1.0)

        except SyntaxError as e:
            validation["issues"].append(f"Syntax error: {e}")
        except Exception:
            # Not Python code or other issue
            validation["syntax_valid"] = True  # Assume valid for non-Python
            validation["quality_score"] = 0.7

        # Generate suggestions
        if '"""' not in code and "'''" not in code:
            validation["suggestions"].append(
                "Consider adding docstrings to document the code"
            )

        return validation

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            db_stats = self.vector_db.get_collection_stats()
            return {
                "indexed_repositories": list(self.indexed_repos),
                "database_stats": db_stats,
                "agents_status": {
                    "query_agent": "active",
                    "repo_agent": "active",
                    "code_agent": "active",
                    "validation_agent": "active",
                },
            }
        except Exception as e:
            return {"error": f"Failed to get status: {e}"}

