"""
Comprehensive API Testing Module
"""
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Optional, Any
import openai==0.28
import pytest
import sys
import os
from pathlib import Path
from config import get_test_config
from utils import calculate_similarity
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy.spatial.distance import cosine
import numpy as np
from bson.objectid import ObjectId

# 加载环境变量
load_dotenv()

# 添加当前目录到 Python 路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Create FastAPI app and test client
app = FastAPI(
    title="Organization Matching API",
    description="API for organization matching and tag generation",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = TestClient(app)

# ----------- Define Structured Tagging Steps -----------
STEP_DESCRIPTIONS = {
    1: os.getenv("TAG_STEP_1"),
    2: os.getenv("TAG_STEP_2"),
    3: os.getenv("TAG_STEP_3"),
    4: os.getenv("TAG_STEP_4"),
    5: os.getenv("TAG_STEP_5"),
    6: os.getenv("TAG_STEP_6")
}

# Data Models
class ProfileRequest(BaseModel):
    description: str
    audience_demographic: str
    additional_info: Dict[str, str]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ProfileResponse(BaseModel):
    tags: List[str]
    embedding: List[float]
    structured_tags: Dict[str, List[str]]
    status: str

class MatchGenerateRequest(BaseModel):
    requirements: str
    type: str = "Non Profit"

class MatchGenerateResponse(BaseModel):
    matches: List[Dict[str, str]]

class MatchRequest(BaseModel):
    requirements: str
    type: str = "Non Profit"
    state: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class MatchResponse(BaseModel):
    matches: List[Dict[str, Any]]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CompanyResponse(BaseModel):
    _id: str
    Name: Optional[str] = None
    Address: Optional[str] = None
    City: Optional[str] = None
    State: Optional[str] = None
    Code: Optional[str] = None
    URL: Optional[str] = None
    Description: Optional[str] = None
    linkedin_url: Optional[str] = None
    linkedin_description: Optional[str] = None
    linkedin_type: Optional[str] = None
    linkedin_logo: Optional[str] = None
    linkedin_cover: Optional[str] = None
    linkedin_staff_count: Optional[str] = None
    linkedin_state: Optional[str] = None
    linkedin_country: Optional[str] = None
    linkedin_city: Optional[str] = None
    linkedin_postal_code: Optional[str] = None
    linkedin_industries: Optional[str] = None
    linkedin_specialities: Optional[str] = None
    linkedin_follower_count: Optional[str] = None
    linkedin_staff_range: Optional[str] = None
    linkedin_crunchbase: Optional[str] = None
    Quality: Optional[str] = None
    Tag: Optional[List[str]] = None
    Embedding: Optional[List[float]] = None
    Assets_USD: Optional[float] = None
    Employees_Total: Optional[int] = None
    Pre_Tax_Profit_USD: Optional[float] = None
    Sales_USD: Optional[float] = None
    linkedin_tagline: Optional[str] = None
    linkedin_phone: Optional[str] = None
    website_scrape: Optional[str] = None
    website_partnership: Optional[str] = None
    website_event: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Add error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)}
    )

# MongoDB 连接
try:
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client[os.getenv("MONGODB_DB_NAME")]  # 现在指向 Organization4
    nonprofit_collection = db[os.getenv("MONGODB_COLLECTION_NONPROFIT")]
    forprofit_collection = db[os.getenv("MONGODB_COLLECTION_FORPROFIT")]
except Exception as e:
    print(f"Database connection error: {e}")

# API Endpoints
@app.get("/")
async def root():
    """测试 API 是否正常运行"""
    return {
        "status": "online",
        "message": "API is running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """检查所有服务的健康状态"""
    status = {
        "api": True,
        "database": False,
        "openai": False
    }
    
    # 检查数据库连接
    try:
        # 直接使用已经初始化的全局数据库连接
        # 测试数据库连接是否活跃
        nonprofit_collection.database.client.server_info()
        status["database"] = True
    except Exception as e:
        print(f"Database connection error in health check: {e}")
        status["database"] = False
    
    # 检查 OpenAI API
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.Model.list()
        status["openai"] = True
    except Exception as e:
        status["openai"] = False
    
    return status

@app.get("/health/database/debug")
async def database_debug():
    """数据库连接调试"""
    try:
        info = {
            "status": "checking",
            "configured_db_name": os.getenv("MONGODB_DB_NAME"),
            "configured_nonprofit_collection": os.getenv("MONGODB_COLLECTION_NONPROFIT"),
            "configured_forprofit_collection": os.getenv("MONGODB_COLLECTION_FORPROFIT"),
            "available_databases": [],
            "current_db_collections": [],
            "collection_counts": {}
        }
        
        # 检查可用的数据库
        info["available_databases"] = client.list_database_names()
        
        # 检查当前数据库的集合
        info["current_db_collections"] = db.list_collection_names()
        
        # 检查集合中的文档数量
        info["collection_counts"] = {
            os.getenv("MONGODB_COLLECTION_NONPROFIT"): nonprofit_collection.count_documents({}),
            os.getenv("MONGODB_COLLECTION_FORPROFIT"): forprofit_collection.count_documents({})
        }
        
        info["status"] = "connected"
        return info
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "configured_db_name": os.getenv("MONGODB_DB_NAME"),
            "configured_nonprofit_collection": os.getenv("MONGODB_COLLECTION_NONPROFIT"),
            "configured_forprofit_collection": os.getenv("MONGODB_COLLECTION_FORPROFIT")
        }

@app.post("/test/generate/tags")
async def generate_tags_api(request: Dict):
    """生成标签的API"""
    try:
        if not request.get("description"):
            raise HTTPException(status_code=400, detail="Missing description")
            
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # 定义所需的变量
        total_tags = 30
        steps = 6
        tags_per_step = 5
        
        # 格式化系统提示
        system_prompt = os.getenv("PROMPT_TAGS_SYSTEM").format(
            total_tags=total_tags,
            steps=steps,
            tags_per_step=tags_per_step
        )
        
        # 格式化用户提示
        user_prompt = os.getenv("PROMPT_TAGS_USER").format(
            total_tags=total_tags,
            description=request["description"]
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        tags = response.choices[0].message['content'].strip()
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        return {
            "status": "success",
            "tags": tag_list[:30],
            "tags_string": ", ".join(tag_list[:30])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/generate/embedding")
async def generate_embedding_api(request: Dict):
    """生成嵌入向量的API"""
    try:
        if not request.get("text"):
            raise HTTPException(status_code=400, detail="Missing text")
            
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[request["text"]]
        )
        
        return {
            "status": "success",
            "embedding": response["data"][0]["embedding"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/generate/ideal-organizations")
async def generate_organizations_api(request: Dict):
    """生成理想组织的API"""
    try:
        required_fields = ["Organization looking 1", "Organization looking 2"]
        if not all(field in request for field in required_fields):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # 生成组织
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": os.getenv("PROMPT_GEN_ORG_SYSTEM").format(
                    org_type_looking_for=request["Organization looking 1"])},
                {"role": "user", "content": os.getenv("PROMPT_GEN_ORG_USER").format(
                    org_type_looking_for=request["Organization looking 1"],
                    partnership_description=request["Organization looking 2"])}
            ]
        )
        
        generated_orgs = response.choices[0].message['content'].strip()
        
        # 过滤组织
        if request.get("Description"):
            filtered_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": os.getenv("PROMPT_FILTER_SYSTEM")},
                    {"role": "user", "content": os.getenv("PROMPT_FILTER_USER").format(
                        organization_mission=request["Description"],
                        generated_organizations=generated_orgs)}
                ]
            )
            return {
                "status": "success",
                "generated_organizations": generated_orgs,
                "filtered_organizations": filtered_response.choices[0].message['content'].strip()
            }
        
        return {
            "status": "success",
            "generated_organizations": generated_orgs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/find-matches")
async def find_matches_api(request: Dict):
    """查找匹配的API"""
    try:
        if not all(key in request for key in ["embedding", "looking_for_type"]):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        matches = []
        collection = nonprofit_collection if request["looking_for_type"].strip() == os.getenv("MONGODB_COLLECTION_NONPROFIT").strip() else forprofit_collection
        
        for org in collection.find({"Embedding": {"$exists": True}}):
            if org.get("Embedding"):
                org_embedding = np.frombuffer(org["Embedding"], dtype=np.float32)
                similarity = 1 - cosine(request["embedding"], org_embedding)
                matches.append((
                    similarity,
                    org.get("Name", "Unknown"),
                    org.get("Description", "No description available"),
                    org.get("URL", "N/A"),
                    org.get("linkedin_description", "No LinkedIn description available"),
                    org.get("linkedin_tagline", "No tagline available"),
                    org.get("linkedin_type", "N/A"),
                    org.get("linkedin_industries", "N/A"),
                    org.get("linkedin_specialities", "N/A"),
                    org.get("linkedin_staff_count", "N/A"),
                    org.get("City", "N/A"),
                    org.get("State", "N/A"),
                    org.get("linkedin_url", "N/A"),
                    org.get("Tag", "No tags available")
                ))
        
        matches.sort(reverse=True)
        return {"matches": matches[:100]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/evaluate-match")
async def evaluate_match_api(request: Dict):
    """评估匹配的API"""
    try:
        if not all(key in request for key in ["match_info", "user_info"]):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        formatted_prompt = os.getenv("MATCH_EVALUATION_PROMPT").format(
            user_type=request["user_info"]["Type"],
            user_description=request["user_info"]["Description"],
            user_target_audience=request["user_info"]["Target Audience"],
            user_looking_type=request["user_info"]["Organization looking 1"],
            user_partnership_desc=request["user_info"]["Organization looking 2"],
            match_name=request["match_info"][1],
            match_description=request["match_info"][2],
            match_linkedin_desc=request["match_info"][4],
            match_tagline=request["match_info"][5],
            match_type=request["match_info"][6],
            match_industry=request["match_info"][7],
            match_specialties=request["match_info"][8],
            match_tags=request["match_info"][13]
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": os.getenv("MATCH_EVALUATION_SYSTEM_PROMPT")},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.3
        )
        
        return {"is_match": response.choices[0]['message']['content'].strip().lower() == 'true'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 高层整合API
@app.post("/test/complete-matching-process")
async def complete_matching_process(request: Dict):
    """完整的匹配流程API"""
    try:
        # 验证输入
        required_fields = [
            "Name", "Type", "Description", "Target Audience",
            "Organization looking 1", "Organization looking 2"
        ]
        if not all(field in request for field in required_fields):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # 1. 生成理想组织
        orgs_response = await generate_organizations_api(request)
        
        # 2. 生成标签
        tags_response = await generate_tags_api({
            "description": orgs_response["filtered_organizations"]
        })
        
        # 3. 生成嵌入向量
        if tags_response["tags_string"]:
            embedding_response = await generate_embedding_api({
                "text": tags_response["tags_string"]
            })
            
            # 4. 查找匹配
            collection = nonprofit_collection if request["Organization looking 1"].strip() == os.getenv("MONGODB_COLLECTION_NONPROFIT").strip() else forprofit_collection
            
            matches = []
            for org in collection.find({"Embedding": {"$exists": True}}):
                if org.get("Embedding"):
                    # 处理 Embedding 字段
                    org_embedding = np.frombuffer(org["Embedding"], dtype=np.float32)
                    similarity = 1 - cosine(embedding_response["embedding"], org_embedding)
                    
                    # 处理数据转换
                    org_dict = dict(org)
                    org_dict["_id"] = str(org_dict["_id"])  # 转换 ObjectId 为字符串
                    
                    # 处理 Tag 字段
                    if isinstance(org_dict.get("Tag"), str):
                        org_dict["Tag"] = org_dict["Tag"].split(", ")
                    elif org_dict.get("Tag") is None:
                        org_dict["Tag"] = []
                    
                    # 处理 Embedding 字段
                    if isinstance(org_dict.get("Embedding"), bytes):
                        org_dict["Embedding"] = np.frombuffer(org_dict["Embedding"], dtype=np.float32).tolist()
                    
                    try:
                        company_response = CompanyResponse(**org_dict)
                        matches.append({
                            "similarity": float(similarity),
                            "organization": company_response.dict()
                        })
                    except Exception as e:
                        print(f"Error processing organization: {str(e)}")
                        continue
            
            # 排序并返回前20个匹配
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "status": "success",
                "suggested_organizations": orgs_response["filtered_organizations"],
                "tags": tags_response["tags_string"],
                "matches": matches[:20]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate tags")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/company/{company_id}")
async def get_company_details(company_id: str):
    """Get company details from database."""
    try:
        # 首先在 nonprofit 集合中查找
        company = nonprofit_collection.find_one({"_id": ObjectId(company_id)})
        if not company:
            # 如果没找到，在 forprofit 集合中查找
            company = forprofit_collection.find_one({"_id": ObjectId(company_id)})
        
        if company:
            # 转换 ObjectId 为字符串
            company["_id"] = str(company["_id"])
            # 转换 Binary Embedding 为列表（如果存在）
            if "Embedding" in company:
                company["Embedding"] = np.frombuffer(company["Embedding"], dtype=np.float32).tolist()
            return company
        else:
            raise HTTPException(status_code=404, detail="Company not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test Functions
def test_profile_processing():
    """Test profile processing"""
    try:
        profile = {
            "description": "Test description",
            "audience_demographic": "Test audience",
            "additional_info": {
                "name": "Test Company",
                "type": "Non Profit",
                "state": "California",
                "city": "Test City",
                "category": "Test Category"
            }
        }
        response = client.post("/test/process/profile", json=profile)
        assert response.status_code == 200
        data = response.json()
        assert "tags" in data
        assert len(data["tags"].split(",")) == 30
    except Exception as e:
        print(f"Profile processing error: {str(e)}")
        raise

def test_generate_embedding():
    """Test embedding generation"""
    try:
        response = client.post(
            "/test/generate/embedding",
            json={
                "text": "Test text"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert "model" in data
    except Exception as e:
        print(f"Embedding generation error: {str(e)}")
        raise

def test_complete_matching():
    """Test complete matching"""
    try:
        request = {
            "embedding": [0.1] * 1536,
            "filters": {
                "type": "Non Profit",
                "state": "California"
            }
        }
        response = client.post("/test/match", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
    except Exception as e:
        print(f"Complete matching error: {str(e)}")
        raise

def test_company_details():
    """Test company details"""
    try:
        company_id = "67ba70f5f8bfb6dcb4492281"
        response = client.get(f"/test/company/{company_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["company_id"] == company_id
    except Exception as e:
        print(f"Company details error: {str(e)}")
        raise

def generate_structured_tags(description: str) -> Dict:
    """生成结构化标签"""
    try:
        tags_by_step = {}
        for step in range(1, 7):
            step_description = os.getenv(f"TAG_STEP_{step}")
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": os.getenv("PROMPT_TAGS_SYSTEM").format(
                        total_tags=5,
                        steps=1,
                        tags_per_step=5
                    )},
                    {"role": "user", "content": f"For the category '{step_description}', {os.getenv('PROMPT_TAGS_USER')}".format(
                        total_tags=5,
                        description=description
                    )}
                ]
            )
            tags = response.choices[0]['message']['content'].strip()
            tags_by_step[f"step_{step}"] = [tag.strip() for tag in tags.split(",") if tag.strip()][:5]
        
        # 合并所有标签
        all_tags = [tag for tags in tags_by_step.values() for tag in tags]
        return {
            "structured_tags": tags_by_step,
            "all_tags": all_tags,
            "tags_string": ", ".join(all_tags)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating structured tags: {str(e)}")

@app.post("/test/filter-organizations")
async def filter_organizations(request: Dict):
    """过滤组织API"""
    try:
        if not all(key in request for key in ["generated_organizations", "organization_mission"]):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": os.getenv("PROMPT_FILTER_SYSTEM")},
                {"role": "user", "content": os.getenv("PROMPT_FILTER_USER").format(
                    organization_mission=request["organization_mission"],
                    generated_organizations=request["generated_organizations"]
                )}
            ]
        )
        
        return {"filtered_organizations": response.choices[0]['message']['content'].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 只保留 search2.py 中使用的环境变量检查
required_env_vars = [
    "OPENAI_API_KEY",
    "MONGODB_URI",
    "MONGODB_DB_NAME",
    "MONGODB_COLLECTION_NONPROFIT",
    "MONGODB_COLLECTION_FORPROFIT",
    "PROMPT_GEN_ORG_SYSTEM",
    "PROMPT_GEN_ORG_USER",
    "PROMPT_FILTER_SYSTEM",
    "PROMPT_FILTER_USER",
    "PROMPT_TAGS_SYSTEM",
    "PROMPT_TAGS_USER",
    "MATCH_EVALUATION_SYSTEM_PROMPT",
    "MATCH_EVALUATION_PROMPT"
]

# 检查环境变量
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

@app.post("/test/analyze/match-reasons")
async def analyze_match_reasons(request: Dict):
    """分析匹配理由的API"""
    try:
        if not all(key in request for key in ["user_org", "match_org"]):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        prompt = f"""
        Based on the following information, explain why these organizations would be good partners:
        
        User Organization:
        - Description: {request["user_org"].get('Description', 'N/A')}
        - Target Audience: {request["user_org"].get('Target Audience', 'N/A')}
        
        Potential Partner:
        - Name: {request["match_org"].get('name', 'N/A')}
        - Description: {request["match_org"].get('description', 'N/A')}
        - Type: {request["match_org"].get('type', 'N/A')}
        - Industries: {request["match_org"].get('industries', 'N/A')}
        - Specialties: {request["match_org"].get('specialities', 'N/A')}
        
        Please provide 2-3 key points about why this would be a good partnership.
        Focus on:
        1. Strategic alignment and shared values
        2. Complementary capabilities and resources
        3. Market and audience synergies
        
        Format your response in clear, concise bullet points.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing organizational partnerships."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            "status": "success",
            "analysis": response.choices[0].message['content'].strip()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/analyze/match-risks")
async def analyze_match_risks(request: Dict):
    """分析匹配风险的API"""
    try:
        if not all(key in request for key in ["user_org", "match_org"]):
            raise HTTPException(status_code=400, detail="Missing required fields")
            
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        prompt = f"""
        Based on the following information, identify potential risks or challenges in this partnership:
        
        User Organization:
        - Description: {request["user_org"].get('Description', 'N/A')}
        - Target Audience: {request["user_org"].get('Target Audience', 'N/A')}
        
        Potential Partner:
        - Name: {request["match_org"].get('name', 'N/A')}
        - Description: {request["match_org"].get('description', 'N/A')}
        - Type: {request["match_org"].get('type', 'N/A')}
        - Industries: {request["match_org"].get('industries', 'N/A')}
        - Specialties: {request["match_org"].get('specialities', 'N/A')}
        
        Please provide 2-3 key points about potential risks or challenges to consider.
        Focus on:
        1. Operational challenges
        2. Market and brand alignment risks
        3. Resource and capability gaps
        
        Format your response in clear, concise bullet points.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in risk analysis for organizational partnerships."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            "status": "success",
            "analysis": response.choices[0].message['content'].strip()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
