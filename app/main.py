# main.py
from fastapi import FastAPI
#config from env
from core.database import DatabaseManager, close_database_manager
from config import settings, setup_logging
import dependencies as dep

class RAGApplication:
    def __init__(self):
        self.app = FastAPI(
            title=settings.app_name,
            description="Distributed RAG system on Raspberry Pi cluster",
            version=settings.app_version,
            debug=settings.debug
        )

        self.setup_routes()
    
    def setup_routes(self):
        
       #@self.on_event("shutdown")
       #async def shutdown_event():
       #    """Clean up database connections on app shutdown"""
       #    await close_database_manager()


        @self.app.get("/")
        async def root():
            return {"message": "RAG Chatbot is running!"}
        

        @self.app.get("/health")
        async def health_check(db: dep.DatabaseDep):
            if db == None:
                return {
                "status":   "degraded",
                "service":  settings.app_name,
                "verison":  settings.app_version,
                "debug":    settings.debug,
                "database": "unavailable",
                "logging_level": settings.log_level
                }
            is_healthy = await db.health_check()
            return {
                "status":   "healthy",
                "service":  settings.app_name,
                "verison":  settings.app_version,
                "debug":    settings.debug,
                "database": "healthy" if is_healthy else "unhealthy",
                "logging_level": settings.log_level
                }
        @self.app.get("/test-deps")
        async def test_dependencies(
            db:dep.DatabaseDep,
            llama:dep.LlamaDep,
            embedding:dep.EmbeddingDep):
            return{"db": db.__class__.__name__,"llama":llama.__class__.__name__,"embedding":embedding.__class__.__name__} 

    def setup_cors(self):
        origins = settings.cors_origins.split(",") if settings.cors_origins == "" else ["*"]
        self.app.add_middleware(
            CORSMiddleware, # type: ignore
            allow_origins=origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods = ["*"],
            allow_headers = ["*"]
        )
    


# Create the application instance
setup_logging()
rag_app = RAGApplication()
app = rag_app.app  # This is what uvicorn needs