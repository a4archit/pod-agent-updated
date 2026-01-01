from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import Optional, List
import uuid


class QdrantTempManager:
    """Manage temporary Qdrant vector databases."""
    
    def __init__(
        self,
        location: str = ":memory:",
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """
        Initialize Qdrant client for temporary storage.
        
        Args:
            location: ":memory:" for in-memory (default), or path for disk-based temp storage
            host: Qdrant server host (if using remote server)
            port: Qdrant server port (if using remote server)
        """
        if host and port:
            # Connect to remote Qdrant server
            self.client = QdrantClient(host=host, port=port)
            self.is_remote = True
        else:
            # Use local in-memory or disk-based storage
            self.client = QdrantClient(location=location)
            self.is_remote = False
        
        self.active_collections = set()
    


    def create_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
        recreate: bool = True
    ) -> str:
        """
        Create a temporary collection in Qdrant.
        
        Args:
            collection_name: Name for the collection (auto-generated if None)
            vector_size: Dimension of vectors (default: 1536 for OpenAI embeddings)
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: If True, delete and recreate if collection exists
            
        Returns:
            Collection name
        """
        # Generate unique collection name if not provided
        if collection_name is None:
            collection_name = f"temp_collection_{uuid.uuid4().hex[:8]}"
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(col.name == collection_name for col in collections)
        
        if exists:
            if recreate:
                print(f"Collection '{collection_name}' exists. Recreating...")
                self.client.delete_collection(collection_name)
            else:
                print(f"Collection '{collection_name}' already exists. Reusing...")
                self.active_collections.add(collection_name)
                return collection_name
        
        # Create new collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            )
        )
        
        self.active_collections.add(collection_name)
        print(f"Created collection: '{collection_name}'")
        
        return collection_name
    




    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a specific collection.
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(collection_name)
            self.active_collections.discard(collection_name)
            print(f"Deleted collection: '{collection_name}'")
            return True
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {e}")
            return False
    



    def cleanup_all(self) -> None:
        """Delete all active collections created by this manager."""
        print(f"Cleaning up {len(self.active_collections)} collections...")
        for collection_name in list(self.active_collections):
            self.delete_collection(collection_name)



    
    def get_client(self) -> QdrantClient:
        """Get the underlying Qdrant client."""
        return self.client
    


    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        collections = self.client.get_collections().collections
        return [col.name for col in collections]







# Standalone function for quick setup
def setup_temp_qdrant(
    collection_name: Optional[str] = None,
    vector_size: int = 1536,
    in_memory: bool = True,
    host: Optional[str] = None,
    port: Optional[int] = None
) -> tuple[QdrantClient, str]:
    """
    Quick setup function for temporary Qdrant database.
    
    Args:
        collection_name: Name for collection (auto-generated if None)
        vector_size: Vector dimension
        in_memory: Use in-memory storage (True) or disk (False)
        host: Remote Qdrant host (optional)
        port: Remote Qdrant port (optional)
        
    Returns:
        Tuple of (QdrantClient, collection_name)
    """
    location = ":memory:" if in_memory else "./temp_qdrant_data"
    
    manager = QdrantTempManager(location=location, host=host, port=port)
    collection = manager.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,
        recreate=True
    )
    
    return manager.get_client(), collection








# Example usage
if __name__ == "__main__":
    # Method 1: Using the manager class
    print("=== Method 1: Using Manager Class ===")
    manager = QdrantTempManager(location=":memory:")
    
    # Create a collection
    collection_name = manager.create_collection(
        collection_name="my_docs",
        vector_size=1536,
        recreate=True
    )
    
    # Get client for operations
    client = manager.get_client()
    print(f"Active collections: {manager.list_collections()}")
    
    # Cleanup when done
    manager.cleanup_all()
    
    print("\n=== Method 2: Quick Setup Function ===")
    # Method 2: Quick setup
    client, collection = setup_temp_qdrant(
        collection_name="quick_setup",
        vector_size=1536,
        in_memory=True
    )
    
    print(f"Ready to use collection: {collection}")