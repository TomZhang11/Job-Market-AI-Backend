from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from time import sleep
from datetime import datetime
from pathlib import Path
import shutil

from utils import NoResultsException, EmptyResumeException, NoSkillsException
from agent import process_query
from skills_recommender import recommend_skills

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://lively-grass-0d661140f.2.azurestaticapps.net"],  # Vite dev server ports
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    web_search: bool = False

class QueryResponse(BaseModel):
    response: str = ""

@app.get("/", response_model=QueryResponse)
def read_root():
    return QueryResponse(response="Job Market AI Backend")

# if results found -> response is returned normally
# if no results found, default values are returned
# if error, exception is raised
@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    try:
        print(f"Processing query: {req.query}, web_search: {req.web_search}") # Log the query
        response = process_query(req.query, req.web_search)
        print(f"Response: {response}") # Log success
        return QueryResponse(response=response)
    except NoResultsException:
        print("No results from vector search")
        return QueryResponse()
    except Exception as e:
        print(f"Error processing query: {str(e)}") # Log errors
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        print("--------------------------------")

@app.post("/test", response_model=QueryResponse)
async def test(req: QueryRequest):
    sleep(3) # simulate waiting on response
    return QueryResponse(
        response=(
            "The recursive alignment of provisional frameworks often necessitates a recalibration of context, "
            "even when the context itself is defined only by the absence of stable referents. Consequently, "
            "each iteration of structural ambiguity reinforces the illusion of coherence, such that the act "
            "of interpretation becomes indistinguishable from the act of fabrication. In this sense, clarity "
            "is not a property of the text but rather an emergent artifact of the reader’s insistence on "
            "discovering patterns where none were initially encoded. Thus, what appears to be explanation may "
            "function instead as the performance of explanation, leaving the boundary between understanding "
            "and misunderstanding deliberately unresolved."
        )
    )

@app.post("/testfail", response_model=QueryResponse)
async def test(req: QueryRequest):
    raise HTTPException(status_code=500, detail="test fail")

@app.post("/upload_resume", response_model=QueryResponse)
async def upload_resume(file: UploadFile = File(...)):
    try:
        print(f"Uploading resume: {file.filename}") # Log the filename
        # find path
        upload_dir = Path(__file__).resolve().parent / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        org_name = Path(file.filename).name if file.filename else "resume.pdf"
        org_path = Path(org_name)
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M")
        filename = f"{org_path.stem}-{timestamp}{org_path.suffix or '.pdf'}"
        path = upload_dir / filename

        # save file
        with path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        print(f"Saved resume to: {path}")

        response = recommend_skills(str(path))
        print(f"Response: {response}") # Log success
        return QueryResponse(response=response)
    except NoResultsException:
        print("No results from vector search")
        return QueryResponse()
    except EmptyResumeException:
        print("Empty resume")
        return QueryResponse(response="could not parse resume")
    except NoSkillsException:
        print("No skills")
        return QueryResponse(response="could not find any skills on your resume")
    except Exception as e:
        print(f"Error processing skills recommendation: {str(e)}") # Log errors
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        print("--------------------------------")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
