"""
Comprehensive API Testing Module
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Optional, Any
import openai
import pytest
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy.spatial.distance import cosine
import numpy as np
from bson.objectid import ObjectId
from string import Template
import requests
import json

# 在文件最开始，import之后
print("Starting database connection check...")
load_dotenv(verbose=True)  # 确保能看到环境变量加载过程

# 打印环境变量值（注意隐藏敏感信息）
print(f"MONGODB_DB_NAME: {os.getenv('MONGODB_DB_NAME')}")
print(f"MONGODB_COLLECTION_NONPROFIT: {os.getenv('MONGODB_COLLECTION_NONPROFIT')}")
print(f"MONGODB_COLLECTION_FORPROFIT: {os.getenv('MONGODB_COLLECTION_FORPROFIT')}")

# 加载环境变量
load_dotenv()

# 添加当前目录到 Python 路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Create FastAPI app
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
    about: Optional[str] = None
    about_us_link: Optional[str] = None
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
    linkedin_tagline: Optional[str] = None
    linkedin_phone: Optional[str] = None
    Quality: Optional[str] = None
    Tag: Optional[List[str]] = None
    Embedding: Optional[List[float]] = None
    
    # 特定类型字段
    contribution: Optional[str] = None  # For-Profit
    csr_page_link: Optional[str] = None  # For-Profit
    partnership: Optional[str] = None  # Non Profit
    website_event: Optional[str] = None  # Non Profit
    website_partnership: Optional[str] = None  # Non Profit
    website_scrape: Optional[str] = None  # Non Profit
    
    class Config:
        populate_by_name = True

class AdCampaignRequest(BaseModel):
    user_org: Dict[str, Any]
    match_org: Dict[str, Any]

class AdCampaignResponse(BaseModel):
    ad_copy: str
    visual_prompt: str
    image: Dict[str, str]

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
            model="gpt-4o-mini",
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
        if "tags" not in request:
            raise HTTPException(status_code=400, detail="Missing tags field")
            
        tags = request["tags"]
        if not tags:
            raise HTTPException(status_code=400, detail="Tags cannot be empty")
            
        print(f"Generating embedding for: {tags}")
            
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=tags  # 直接使用字符串
        )
        
        return {
            "status": "success",
            "embedding": response["data"][0]["embedding"]
        }
    except Exception as e:
        print(f"Error: {str(e)}")
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
                    org.get("Tag", "No tags available"),
                    str(org.get("_id"))
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
@app.post("/test/complete-matching-process-new")
async def complete_matching_process(request: Dict):
    """完整的匹配流程API"""
    try:
        # 1. 验证输入
        required_fields = [
            "Name", "Type", "Description", "Target Audience",
            "Organization looking 1", "Organization looking 2"
        ]
        if not all(field in request for field in required_fields):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # 2. 生成理想组织 (/test/generate/ideal-organizations)
        orgs_response = await generate_organizations_api({
            "Name": request["Name"],
            "Type": request["Type"],
            "Description": request["Description"],
            "Organization looking 1": request["Organization looking 1"],
            "Organization looking 2": request["Organization looking 2"]
        })
        
        if not orgs_response.get("filtered_organizations"):
            raise HTTPException(status_code=500, detail="Failed to generate organizations")

        # 3. 生成标签 (/test/generate/tags)
        tags_response = await generate_tags_api({
            "description": orgs_response["filtered_organizations"]
        })
        
        if not tags_response.get("tags"):
            raise HTTPException(status_code=500, detail="Failed to generate tags")

        # 4. 生成嵌入向量 (/test/generate/embedding)
        embedding_response = await generate_embedding_api({
            "tags": tags_response["tags_string"]
        })
        
        if not embedding_response.get("embedding"):
            raise HTTPException(status_code=500, detail="Failed to generate embedding")

        # 5. 查找匹配 (/test/find-matches)
        matches_response = await find_matches_api({
            "embedding": embedding_response["embedding"],
            "looking_for_type": request["Organization looking 1"]
        })

        if not matches_response.get("matches"):
            raise HTTPException(status_code=500, detail="No matches found")

        # 6. 评估匹配
        evaluated_matches = []  # 存储已评估且匹配的
        unmatched = []  # 存储已评估但不匹配的
        unevaluated_matches = []  # 存储剩余未评估的80个
        final_matches = []
        
        # 处理前20个匹配
        first_twenty = matches_response["matches"][:20]
        remaining_matches = matches_response["matches"][20:]
        
        for match in first_twenty:
            try:
                # 1. 验证和转换ObjectId - 与get_company_details相同
                try:
                    object_id = ObjectId(match[14])
                except Exception as e:
                    print(f"Invalid ObjectId format: {str(e)}")
                    continue

                # 2. 在两个集合中查找 - 与get_company_details相同
                org = nonprofit_collection.find_one({"_id": object_id})
                collection_type = "Non Profit"
                
                if not org:
                    org = forprofit_collection.find_one({"_id": object_id})
                    collection_type = "For-Profit"
                
                if not org:
                    print(f"Company not found for id: {object_id}")
                    continue

                # 3. 构建匹配结果
                match_result = {
                    "similarity_score": float(match[0]),
                    "organization": {
                        # 基础信息 - 使用MongoDB中的确切字段名
                        "Name": org.get("Name"),
                        "Description": org.get("Description"),
                        "Address": org.get("Address"),
                        "City": org.get("City"),
                        "State": org.get("State"),
                        "Code": org.get("Code"),
                        "URL": org.get("URL"),
                        
                        # LinkedIn信息 - 保持原有字段名
                        "linkedin_url": org.get("linkedin_url"),
                        "linkedin_tagline": org.get("linkedin_tagline"),
                        "linkedin_industries": org.get("linkedin_industries"),
                        "linkedin_specialities": org.get("linkedin_specialities"),
                        "linkedin_staff_range": org.get("linkedin_staff_range"),
                        "linkedin_follower_count": org.get("linkedin_follower_count"),
                        "linkedin_logo": org.get("linkedin_logo"),
                        "linkedin_crunchbase": org.get("linkedin_crunchbase"),
                        
                        # 组织特定字段 - 保持原有字段名
                        **({"contribution": org.get("contribution"),
                            "csr_page_link": org.get("csr_page_link")} 
                           if collection_type == "For-Profit" else
                           {"partnership": org.get("partnership"),
                            "event": org.get("event"),
                            "website_event": org.get("website_event"),
                            "website_partnership": org.get("website_partnership")})
                    }
                }
                unevaluated_matches.append(match_result)
                
            except Exception as e:
                print(f"Error processing unevaluated match: {str(e)}")
                continue

        # 构建最终的20个匹配
        # 首先添加所有已评估且匹配的
        final_matches = evaluated_matches.copy()
        # 如果不够20个，从未评估的中按相似度补充
        if len(final_matches) < 20:
            remaining_needed = 20 - len(final_matches)
            final_matches.extend(unevaluated_matches[:remaining_needed])

        matching_results = {
            "successful_matches": [
                {
                    "similarity_score": match.get("similarity_score"),
                    "organization": {
                        # 基础信息 - 使用MongoDB中的确切字段名
                        "Name": match.get("organization", {}).get("Name"),
                        "Description": match.get("organization", {}).get("Description"),
                        "Address": match.get("organization", {}).get("Address"),
                        "City": match.get("organization", {}).get("City"),
                        "State": match.get("organization", {}).get("State"),
                        "Code": match.get("organization", {}).get("Code"),
                        "URL": match.get("organization", {}).get("URL"),
                        
                        # LinkedIn信息 - 保持原有字段名
                        "linkedin_url": match.get("organization", {}).get("linkedin_url"),
                        "linkedin_tagline": match.get("organization", {}).get("linkedin_tagline"),
                        "linkedin_industries": match.get("organization", {}).get("linkedin_industries"),
                        "linkedin_specialities": match.get("organization", {}).get("linkedin_specialities"),
                        "linkedin_staff_range": match.get("organization", {}).get("linkedin_staff_range"),
                        "linkedin_follower_count": match.get("organization", {}).get("linkedin_follower_count"),
                        "linkedin_logo": match.get("organization", {}).get("linkedin_logo"),
                        "linkedin_crunchbase": match.get("organization", {}).get("linkedin_crunchbase"),
                        
                        # 组织特定字段 - 保持原有字段名
                        **({"contribution": match.get("organization", {}).get("contribution"),
                            "csr_page_link": match.get("organization", {}).get("csr_page_link")} 
                           if match.get("organization", {}).get("type") == "For-Profit" else
                           {"partnership": match.get("organization", {}).get("partnership"),
                            "event": match.get("organization", {}).get("event"),
                            "website_event": match.get("organization", {}).get("website_event"),
                            "website_partnership": match.get("organization", {}).get("website_partnership")})
                    },
                    "evaluation": {
                        "is_match": match.get("evaluation", {}).get("is_match", True)
                    }
                }
                for match in final_matches
            ]
        }
        
        response = {
            "status": "success",
            "process_steps": {
                "step1_input_organization": {
                    "name": request["Name"],
                    "type": request["Type"],
                    "description": request["Description"],
                    "target_audience": request["Target Audience"],
                    "looking_for": request["Organization looking 1"],
                    "partnership_description": request["Organization looking 2"]
                },
                "step2_suggested_organizations": {
                    "generated": orgs_response["generated_organizations"],
                    "filtered": orgs_response["filtered_organizations"]
                },
                "step3_generated_tags": {
                    "tags": tags_response["tags"],
                    "tags_string": tags_response["tags_string"]
                },
                "step4_embedding": {
                    "dimension": len(embedding_response["embedding"]),
                    "embedding_sample": embedding_response["embedding"][:5]
                },
                "step5_initial_matches": {
                    "total_matches_found": len(matches_response["matches"]),
                    "matches_processed": 20
                },
                "step6_match_evaluation": {
                    "total_matches_found": len(matches_response["matches"]),
                    "evaluated_count": 20,
                    "matched_count": len(evaluated_matches),
                    "unmatched_count": len(unmatched),
                    "unevaluated_count": 80,
                    "final_matches_count": 20
                }
            },
            "matching_results": matching_results
        }
        
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "step": "complete_matching_process",
                "message": "Error in matching process"
            }
        )
        
# 高层整合API
@app.post("/test/complete-matching-process")
async def complete_matching_process(request: Dict):
    """完整的匹配流程API"""
    try:
        # 1. 验证输入
        required_fields = [
            "Name", "Type", "Description", "Target Audience",
            "Organization looking 1", "Organization looking 2"
        ]
        if not all(field in request for field in required_fields):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # 2. 生成理想组织 (/test/generate/ideal-organizations)
        orgs_response = await generate_organizations_api({
            "Name": request["Name"],
            "Type": request["Type"],
            "Description": request["Description"],
            "Organization looking 1": request["Organization looking 1"],
            "Organization looking 2": request["Organization looking 2"]
        })
        
        if not orgs_response.get("filtered_organizations"):
            raise HTTPException(status_code=500, detail="Failed to generate organizations")

        # 3. 生成标签 (/test/generate/tags)
        tags_response = await generate_tags_api({
            "description": orgs_response["filtered_organizations"]
        })
        
        if not tags_response.get("tags"):
            raise HTTPException(status_code=500, detail="Failed to generate tags")

        # 4. 生成嵌入向量 (/test/generate/embedding)
        embedding_response = await generate_embedding_api({
            "tags": tags_response["tags_string"]
        })
        
        if not embedding_response.get("embedding"):
            raise HTTPException(status_code=500, detail="Failed to generate embedding")

        # 5. 查找匹配 (/test/find-matches)
        matches_response = await find_matches_api({
            "embedding": embedding_response["embedding"],
            "looking_for_type": request["Organization looking 1"]
        })

        if not matches_response.get("matches"):
            raise HTTPException(status_code=500, detail="No matches found")

        # 6. 评估匹配
        evaluated_matches = []  # 存储已评估且匹配的
        unmatched = []  # 存储已评估但不匹配的
        unevaluated_matches = []  # 存储剩余未评估的80个
        final_matches = []
        
        # 处理前20个匹配
        first_twenty = matches_response["matches"][:20]
        remaining_matches = matches_response["matches"][20:]
        
        for match in first_twenty:
            try:
                # 1. 验证和转换ObjectId - 与get_company_details相同
                try:
                    object_id = ObjectId(match[14])
                except Exception as e:
                    print(f"Invalid ObjectId format: {str(e)}")
                    continue

                # 2. 在两个集合中查找 - 与get_company_details相同
                org = nonprofit_collection.find_one({"_id": object_id})
                collection_type = "Non Profit"
                
                if not org:
                    org = forprofit_collection.find_one({"_id": object_id})
                    collection_type = "For-Profit"
                
                if not org:
                    print(f"Company not found for id: {object_id}")
                    continue

                # 3. 构建匹配结果
                match_result = {
                    "similarity_score": float(match[0]),
                    "organization": {
                        # 基础信息 - 使用MongoDB中的确切字段名
                        "Name": org.get("Name"),
                        "Description": org.get("Description"),
                        "Address": org.get("Address"),
                        "City": org.get("City"),
                        "State": org.get("State"),
                        "Code": org.get("Code"),
                        "URL": org.get("URL"),
                        
                        # LinkedIn信息 - 保持原有字段名
                        "linkedin_url": org.get("linkedin_url"),
                        "linkedin_tagline": org.get("linkedin_tagline"),
                        "linkedin_industries": org.get("linkedin_industries"),
                        "linkedin_specialities": org.get("linkedin_specialities"),
                        "linkedin_staff_range": org.get("linkedin_staff_range"),
                        "linkedin_follower_count": org.get("linkedin_follower_count"),
                        "linkedin_logo": org.get("linkedin_logo"),
                        "linkedin_crunchbase": org.get("linkedin_crunchbase"),
                        
                        # 组织特定字段 - 保持原有字段名
                        **({"contribution": org.get("contribution"),
                            "csr_page_link": org.get("csr_page_link")} 
                           if collection_type == "For-Profit" else
                           {"partnership": org.get("partnership"),
                            "event": org.get("event"),
                            "website_event": org.get("website_event"),
                            "website_partnership": org.get("website_partnership")})
                    }
                }
                unevaluated_matches.append(match_result)
                
            except Exception as e:
                print(f"Error processing unevaluated match: {str(e)}")
                continue

        # 构建最终的20个匹配
        # 首先添加所有已评估且匹配的
        final_matches = evaluated_matches.copy()
        # 如果不够20个，从未评估的中按相似度补充
        if len(final_matches) < 20:
            remaining_needed = 20 - len(final_matches)
            final_matches.extend(unevaluated_matches[:remaining_needed])

        matching_results = {
            "successful_matches": [
                {
                    "similarity_score": match.get("similarity_score"),
                    "organization": {
                        # 基础信息 - 使用MongoDB中的确切字段名
                        "Name": match.get("organization", {}).get("Name"),
                        "Description": match.get("organization", {}).get("Description"),
                        "Address": match.get("organization", {}).get("Address"),
                        "City": match.get("organization", {}).get("City"),
                        "State": match.get("organization", {}).get("State"),
                        "Code": match.get("organization", {}).get("Code"),
                        "URL": match.get("organization", {}).get("URL"),
                        
                        # LinkedIn信息 - 保持原有字段名
                        "linkedin_url": match.get("organization", {}).get("linkedin_url"),
                        "linkedin_tagline": match.get("organization", {}).get("linkedin_tagline"),
                        "linkedin_industries": match.get("organization", {}).get("linkedin_industries"),
                        "linkedin_specialities": match.get("organization", {}).get("linkedin_specialities"),
                        "linkedin_staff_range": match.get("organization", {}).get("linkedin_staff_range"),
                        "linkedin_follower_count": match.get("organization", {}).get("linkedin_follower_count"),
                        "linkedin_logo": match.get("organization", {}).get("linkedin_logo"),
                        "linkedin_crunchbase": match.get("organization", {}).get("linkedin_crunchbase"),
                        
                        # 组织特定字段 - 保持原有字段名
                        **({"contribution": match.get("organization", {}).get("contribution"),
                            "csr_page_link": match.get("organization", {}).get("csr_page_link")} 
                           if match.get("organization", {}).get("type") == "For-Profit" else
                           {"partnership": match.get("organization", {}).get("partnership"),
                            "event": match.get("organization", {}).get("event"),
                            "website_event": match.get("organization", {}).get("website_event"),
                            "website_partnership": match.get("organization", {}).get("website_partnership")})
                    },
                    "evaluation": {
                        "is_match": match.get("evaluation", {}).get("is_match", True)
                    }
                }
                for match in final_matches
            ]
        }
        
        response = {
            "status": "success",
            "process_steps": {
                "step1_input_organization": {
                    "name": request["Name"],
                    "type": request["Type"],
                    "description": request["Description"],
                    "target_audience": request["Target Audience"],
                    "looking_for": request["Organization looking 1"],
                    "partnership_description": request["Organization looking 2"]
                },
                "step2_suggested_organizations": {
                    "generated": orgs_response["generated_organizations"],
                    "filtered": orgs_response["filtered_organizations"]
                },
                "step3_generated_tags": {
                    "tags": tags_response["tags"],
                    "tags_string": tags_response["tags_string"]
                },
                "step4_embedding": {
                    "dimension": len(embedding_response["embedding"]),
                    "embedding_sample": embedding_response["embedding"][:5]
                },
                "step5_initial_matches": {
                    "total_matches_found": len(matches_response["matches"]),
                    "matches_processed": 20
                },
                "step6_match_evaluation": {
                    "total_matches_found": len(matches_response["matches"]),
                    "evaluated_count": 20,
                    "matched_count": len(evaluated_matches),
                    "unmatched_count": len(unmatched),
                    "unevaluated_count": 80,
                    "final_matches_count": 20
                }
            },
            "matching_results": matching_results
        }
        
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "step": "complete_matching_process",
                "message": "Error in matching process"
            }
        )


@app.get("/test/company/{company_id}")
async def get_company_details(company_id: str):
    """Get company details from database."""
    try:
        # 直接验证和转换为 ObjectId
        try:
            object_id = ObjectId(company_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid ObjectId format. Must be a 24-character hex string."
            )

        # 在两个集合中查找
        company = nonprofit_collection.find_one({"_id": object_id})
        collection_type = "Non Profit"
        
        if not company:
            company = forprofit_collection.find_one({"_id": object_id})
            collection_type = "For-Profit"
        
        if company:
            # 转换 ObjectId 为字符串
            company["_id"] = str(company["_id"])
            company["organization_type"] = collection_type
            
            # 转换 Embedding 如果存在
            if "Embedding" in company:
                company["Embedding"] = np.frombuffer(company["Embedding"], dtype=np.float32).tolist()
            return company
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Company not found"
            )
            
    except HTTPException as he:
        raise he
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
    example_data = {
        "Name": "Healthy Food Co",
        "Type": "For-Profit",
        "Description": "We provide organic and healthy food solutions",
        "Target Audience": "Health-conscious consumers",
        "Organization looking 1": "Non Profit",
        "Organization looking 2": "Looking for partnership with health education organizations"
    }

    # 定义字段映射关系
    field_mapping = {
        # 通用字段
        "common_fields": {
            "Name": "name",
            "Description": "description",
            "URL": "url",
            "City": "city",
            "State": "state",
            "Address": "address",
            "Code": "code",
            "linkedin_description": "linkedin_description",
            "linkedin_follower_count": "follower_count",
            "linkedin_industries": "industries",
            "linkedin_logo": "logo",
            "linkedin_specialities": "specialities",
            "linkedin_staff_range": "staff_range",
            "linkedin_tagline": "tagline",
            "linkedin_type": "type",
            "linkedin_url": "linkedin_url",
            "Assets (USD)": "assets",
            "Employees (Total)": "employees",
            "Pre Tax Profit (USD)": "pre_tax_profit",
            "Sales (USD)": "sales",
            "_id": "_id"
        },
        # 营利组织特有字段
        "for_profit_only": {
            "contribution": "contribution",
            "csr_page_link": "csr_page_link"
        },
        # 非营利组织特有字段
        "non_profit_only": {
            "partnership": "partnership",
            "website_event": "website_event",
            "website_partnership": "website_partnership",
            "event": "event"
        }
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/test/complete-matching-process",
            headers={"Content-Type": "application/json"},
            json=example_data
        )
        
        data = response.json()
        
        # 分析每个匹配的组织
        print("\nAnalyzing matches:")
        for match in data.get("matches", []):
            org = match.get("organization", {})
            org_type = org.get("type")
            
            print(f"\nOrganization: {org.get('name')}")
            print(f"Type: {org_type}")
            print(f"Similarity Score: {match.get('similarity_score')}")
            
            # 检查通用字段
            print("\nCommon Fields:")
            for db_field, api_field in field_mapping["common_fields"].items():
                value = org.get(api_field)
                print(f"{db_field}: {'Present' if value is not None else 'Missing'}")
            
            # 检查类型特定字段
            if org_type == "For-Profit":
                print("\nFor-Profit Specific Fields:")
                for db_field, api_field in field_mapping["for_profit_only"].items():
                    value = org.get(api_field)
                    print(f"{db_field}: {'Present' if value is not None else 'Missing'}")
            else:
                print("\nNon Profit Specific Fields:")
                for db_field, api_field in field_mapping["non_profit_only"].items():
                    value = org.get(api_field)
                    print(f"{db_field}: {'Present' if value is not None else 'Missing'}")
            
            # 统计字段覆盖率
            total_fields = len(field_mapping["common_fields"])
            present_fields = sum(1 for field in field_mapping["common_fields"].values() if org.get(field) is not None)
            
            if org_type == "For-Profit":
                total_fields += len(field_mapping["for_profit_only"])
                present_fields += sum(1 for field in field_mapping["for_profit_only"].values() if org.get(field) is not None)
            else:
                total_fields += len(field_mapping["non_profit_only"])
                present_fields += sum(1 for field in field_mapping["non_profit_only"].values() if org.get(field) is not None)
            
            print(f"\nField Coverage: {present_fields}/{total_fields} ({(present_fields/total_fields)*100:.2f}%)")
        
        return data
    except Exception as error:
        print("Error:", error)
        return None

def generate_structured_tags(description: str) -> Dict:
    """生成结构化标签"""
    try:
        tags_by_step = {}
        for step in range(1, 7):
            step_description = os.getenv(f"TAG_STEP_{step}")
            response = openai.ChatCompletion.create(
                model="gpt-4",
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


@app.post("/test/analyze/match-reasons")
async def analyze_match_reasons(request: Dict):
    """Generate detailed matching analysis report in three stages"""
    try:
        # 基础验证
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # 获取组织数据
        match_org_id = ObjectId(request["match_org"].get("_id"))
        detailed_org = nonprofit_collection.find_one({"_id": match_org_id})
        org_type = "Non-Profit"
        
        if not detailed_org:
            detailed_org = forprofit_collection.find_one({"_id": match_org_id})
            org_type = "For-Profit"
            
        if not detailed_org:
            raise HTTPException(status_code=404, detail="Organization not found")

        # Stage 1: Value Alignment Analysis
        alignment_analysis_prompt = f"""
        Analyze value alignment between {detailed_org.get('Name')} and {request['user_org'].get('Name')}:

        User Organization Information:
        - Name: {request['user_org'].get('Name')}
        - Type: {request['user_org'].get('Type')}
        - Description: {request['user_org'].get('Description', '')}
        - Target Audience: {request['user_org'].get('Target Audience', '')}
        - Partnership Goals: {request['user_org'].get('Organization looking 2', '')}

        Matching Organization Identity:
        - About: {detailed_org.get('about', '')}
        - LinkedIn Description: {detailed_org.get('linkedin_description', '')}
        - LinkedIn Tagline: {detailed_org.get('linkedin_tagline', '')}
        - Industries: {detailed_org.get('linkedin_industries', '')}
        - Specialties: {detailed_org.get('linkedin_specialities', '')}
        - Description: {detailed_org.get('Description', '')}

        Provide analysis in JSON format focusing on:
        1. Strategic alignment of missions
        2. Value proposition compatibility
        3. Market positioning alignment
        """

        # Stage 2: Activity Pattern Analysis
        activity_analysis_prompt = f"""
        Analyze activity patterns and focus areas:

        User Organization Profile:
        - Description: {request['user_org'].get('Description', '')}
        - Target Audience: {request['user_org'].get('Target Audience', '')}
        - Partnership Goals: {request['user_org'].get('Organization looking 2', '')}

        Matching Organization Activities:
        {'Corporate Social Responsibility:' if org_type == "For-Profit" else 'Partnership History:'}
        {detailed_org.get('contribution', '') if org_type == "For-Profit" else detailed_org.get('partnership', '')}

        Activity Records:
        - Events: {detailed_org.get('event', '')}
        - Partnership Programs: {detailed_org.get('website_partnership', '')}
        - Event Programs: {detailed_org.get('website_event', '')}
        - CSR Programs: {detailed_org.get('csr_page_link', '')}

        Focus Areas:
        - Industries: {detailed_org.get('linkedin_industries', '')}
        - Specialties: {detailed_org.get('linkedin_specialities', '')}
        - Description: {detailed_org.get('Description', '')}

        Provide JSON analysis of:
        1. Primary Focus Areas
        2. Partnership Style
        3. Activity Patterns
        """

        # Stage 3: Capability Assessment
        capability_analysis_prompt = f"""
        Analyze organizational capabilities:

        User Organization Requirements:
        - Type: {request['user_org'].get('Type', '')}
        - Target Audience: {request['user_org'].get('Target Audience', '')}
        - Partnership Goals: {request['user_org'].get('Organization looking 2', '')}

        Matching Organization Metrics:
        Scale Metrics:
        - Staff Count: {detailed_org.get('linkedin_staff_count', '')}
        - Staff Range: {detailed_org.get('linkedin_staff_range', '')}
        - Total Employees: {detailed_org.get('Employees (Total)', '')}
        - Location: {detailed_org.get('City', '')}, {detailed_org.get('State', '')}

        Financial Indicators:
        - Annual Sales: {detailed_org.get('Sales (USD)', '')}
        - Assets: {detailed_org.get('Assets (USD)', '')}
        - Pre Tax Profit: {detailed_org.get('Pre Tax Profit (USD)', '')}
        - Market Presence: {detailed_org.get('linkedin_follower_count', '')} LinkedIn followers

        Provide JSON analysis of:
        1. Resource Capacity
        2. Implementation Capability
        3. Partnership Readiness
        """
        
        # 执行三阶段分析
        try:
            # 阶段1：价值匹配分析
            alignment_response = await get_openai_analysis(
                alignment_analysis_prompt,
                "You are an expert in analyzing organizational value alignment."
            )

            # 阶段2：活动模式分析
            activity_response = await get_openai_analysis(
                activity_analysis_prompt,
                "You are an expert in analyzing organizational activities and patterns."
            )

            # 阶段3：能力评估
            capability_response = await get_openai_analysis(
                capability_analysis_prompt,
                "You are an expert in assessing organizational capabilities."
            )

            # 返回完整分析
            return {
                "status": "success",
                "match_analysis": {
                    "value_alignment": alignment_response,
                    "activity_pattern": activity_response,
                    "capability_assessment": capability_response
                }
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Analysis error: {str(e)}"
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

async def get_openai_analysis(prompt: str, system_role: str) -> dict:
    """Helper function for OpenAI API calls"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ]
        )
        return json.loads(response.choices[0].message['content'])
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse analysis response"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI API error: {str(e)}"
        )
@app.post("/test/generate/ad-campaign", response_model=AdCampaignResponse)
async def generate_ad_campaign(request: AdCampaignRequest):
    """生成广告活动的API"""
    try:
        # 设置OpenAI API key
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # 打印用户输入数据
        print("User Organization Input:")
        print(json.dumps(request.user_org, indent=2))
        
        # 获取匹配组织
        match_org_id = request.match_org.get("_id")
        try:
            object_id = ObjectId(match_org_id)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid ObjectId format"
            )

        # 在数据库中查找
        match_org = nonprofit_collection.find_one({"_id": object_id})
        collection_type = "Non Profit"
        
        if not match_org:
            match_org = forprofit_collection.find_one({"_id": object_id})
            collection_type = "For-Profit"
            
        if not match_org:
            raise HTTPException(
                status_code=404,
                detail=f"Company not found"
            )
            
        # 打印数据库获取的组织数据
        print("\nMatched Organization from Database:")
        print(json.dumps({
            "Name": match_org.get("Name"),
            "Description": match_org.get("Description"),
            "linkedin_industries": match_org.get("linkedin_industries"),
            "linkedin_specialities": match_org.get("linkedin_specialities"),
            "linkedin_description": match_org.get("linkedin_description"),
            "linkedin_tagline": match_org.get("linkedin_tagline"),
            "type": collection_type
        }, indent=2))

        # 生成广告文案
        ad_copy = generate_ad_copy(request.user_org, match_org)
        print("\nGenerated Ad Copy:")
        print(ad_copy)
        
        if not ad_copy or ad_copy == "Unable to generate ad copy at this time.":
            raise HTTPException(status_code=500, detail="Failed to generate ad copy")
            
        # 生成视觉提示
        visual_prompt = generate_visual_prompt(ad_copy, request.user_org, match_org)
        print("\nGenerated Visual Prompt:")
        print(visual_prompt)
        
        if not visual_prompt or visual_prompt == "Unable to generate visual recommendations at this time.":
            raise HTTPException(status_code=500, detail="Failed to generate visual prompt")
            
        # 生成图片
        image_result = generate_image(ad_copy, request.user_org, match_org)
        if 'error' in image_result:
            raise HTTPException(status_code=500, detail=f"Failed to generate image")
            
        return AdCampaignResponse(
            ad_copy=ad_copy,
            visual_prompt=visual_prompt,
            image=image_result
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in generate_ad_campaign: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_completion(prompt, is_visual=False):
    """调用OpenAI API获取回复"""
    try:
        system_content = os.getenv("AD_SYSTEM_ROLE")
        if is_visual:
            system_content += "\nIMPORTANT: You MUST use the exact headline, description, and visual style provided in the prompt. Do not create new content. Only provide layout and design instructions for the given content."
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "I will only use the provided content and ensure all variables are properly replaced."}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        content = response.choices[0].message['content']
        
        # 如果是视觉提示，验证是否包含必需的部分
        if is_visual and not all(section in content for section in [
            "### Layout Composition and Logo Placement",
            "### Color Palette and Mood",
            "### Typography",
            "### Visual Elements and Imagery",
            "### Visual Hierarchy"
        ]):
            raise ValueError("Response missing required sections")
            
        return content
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return "Unable to generate content at this time."

def generate_ad_copy(user_org, match_org):
    """Generate AI-based ad copy for partnership announcement"""
    try:
        # 确保所有字段都有值，如果不存在则使用空字符串
        required_fields = {
            "user_org_name": user_org.get("Name", ""),
            "user_org_type": user_org.get("Type", ""),
            "user_org_desc": user_org.get("Description", ""),
            "match_org_name": match_org.get("Name", ""),
            "match_org_type": match_org.get("type", ""),
            "match_org_industry": match_org.get("linkedin_industries", ""),
            "match_org_specialties": match_org.get("linkedin_specialities", ""),
            "match_org_linkedin_desc": match_org.get("linkedin_description", ""),
            "match_org_tagline": match_org.get("linkedin_tagline", ""),
            "audience": user_org.get("Target Audience", "")
        }
        
        # 打印调试信息
        print("Debug - Template variables before replacement:")
        print(json.dumps(required_fields, indent=2))
        
        # 使用模板替换
        prompt_template = Template(os.getenv("PARTNERSHIP_AD_COPY_PROMPT"))
        prompt = prompt_template.safe_substitute(required_fields)
        
        # 添加额外的提醒
        prompt = f"""IMPORTANT: Use EXACTLY these organization names:
1. User Organization: {required_fields['user_org_name']}
2. Partner Organization: {required_fields['match_org_name']}

{prompt}"""
        
        response = get_completion(prompt)
        return response
        
    except Exception as e:
        print(f"Error generating ad copy: {str(e)}")
        return "Unable to generate ad copy at this time."

def generate_visual_prompt(ad_copy, user_org, match_org):
    """Generate visual prompt based on ad copy and partnership details"""
    try:
        # 解析广告文案中的各个部分
        sections = {
            'headline': '',
            'description': '',
            'visual_style': ''
        }
        
        current_section = None
        for line in ad_copy.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('HEADLINE:'):
                current_section = 'headline'
                sections['headline'] = line.replace('HEADLINE:', '').strip()
            elif line.startswith('DESCRIPTION:'):
                current_section = 'description'
                sections['description'] = line.replace('DESCRIPTION:', '').strip()
            elif line.startswith('VISUAL_STYLE:'):
                current_section = 'visual_style'
                sections['visual_style'] = line.replace('VISUAL_STYLE:', '').strip()
            elif current_section and not line.startswith('CTA_TAGS:'):
                sections[current_section] += ' ' + line
        
        # 准备模板变量
        template_vars = {
            "headline": sections['headline'],
            "description": sections['description'],
            "visual_style": sections['visual_style'],
            "user_org_name": user_org.get("Name", ""),
            "user_org_type": user_org.get("Type", ""),
            "user_org_desc": user_org.get("Description", ""),
            "match_org_name": match_org.get("Name", ""),
            "match_org_type": match_org.get("type", "For-Profit"),
            "match_org_industries": match_org.get("linkedin_industries", ""),
            "match_org_specialties": match_org.get("linkedin_specialities", ""),
            "match_org_tagline": match_org.get("linkedin_tagline", "")
        }

        # 使用Template进行替换
        prompt_template = Template(os.getenv("VISUAL_PROMPT_TEMPLATE"))
        prompt = prompt_template.safe_substitute(template_vars)
        
        response = get_completion(prompt, is_visual=True)
        
        # 确保所有变量都被替换
        final_response = response
        for key, value in template_vars.items():
            final_response = final_response.replace(f"{{{key}}}", value)
            final_response = final_response.replace(f'"{{{key}}}"', f'"{value}"')
            final_response = final_response.replace(f"[{key}]", value)
            
        # 特别处理引号内的变量
        final_response = final_response.replace('"{headline}"', f'"{sections["headline"]}"')
        final_response = final_response.replace('"{description}"', f'"{sections["description"]}"')
        final_response = final_response.replace('{visual_style}', sections['visual_style'])
        
        return final_response
        
    except Exception as e:
        print(f"Error generating visual prompt: {str(e)}")
        return "Unable to generate visual recommendations at this time."

def generate_image(ad_copy, user_org, match_org):
    """Generate image using Ideogram API"""
    try:
        api_key = os.getenv("IDEOGRAM_API_KEY")
        if not api_key:
            raise ValueError("Ideogram API key not found in environment variables")
            
        response = requests.post(
            os.getenv("IDEOGRAM_API_ENDPOINT"),
            headers={
                "Api-Key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "image_request": {
                    "prompt": generate_visual_prompt(ad_copy, user_org, match_org),
                    "aspect_ratio": os.getenv("IDEOGRAM_IMAGE_ASPECT_RATIO"),
                    "model": os.getenv("IDEOGRAM_MODEL_VERSION"),
                    "magic_prompt_option": "ON",
                    "color_palette": {
                        "name": os.getenv("IDEOGRAM_COLOR_PALETTE")
                    }
                }
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
        response_data = response.json()
        if not response_data.get('data') or not response_data['data'][0].get('url'):
            raise Exception("No image URL found in API response")
            
        return {"url": response_data['data'][0]['url']}
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return {"error": str(e)}

def safe_float_conversion(value):
    """安全地转换数值，处理特殊情况"""
    if isinstance(value, dict) and "$numberDouble" in value:
        try:
            float_val = float(value["$numberDouble"])
            # 检查是否是 NaN
            if np.isnan(float_val):
                return None
            return float_val
        except (ValueError, TypeError):
            return None
    return None

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
    
