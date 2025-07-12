from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from typing import List, Dict
from pydantic import BaseModel, Field
import numpy as np
from sklearn.cluster import KMeans
import pymap3d as pm 

class Place(BaseModel):
    name: str
    address: str
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)

class ClusterRequest(BaseModel):
    group: int = Field(..., ge=1, description="나눌 그룹 수 설정")
    place: List[Place] = Field(..., min_items=1)

def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    x, y, z = pm.geodetic2ecef(lat_deg, lon_deg, np.zeros_like(lat_deg))
    return np.column_stack([x, y, z])

def run_kmeans(coords_deg: np.ndarray, k: int = 5) -> np.ndarray:
    """coords_deg: (N, 2) lat/lon → k-means labels."""
    if len(coords_deg) < k:
        raise ValueError(f"K({k}) must be ≤ number of points({len(coords_deg)})")
    xyz = geodetic_to_ecef(coords_deg[:, 0], coords_deg[:, 1])
    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(xyz)
    return labels


app = FastAPI()

origins = [
    "https://store-group-six.vercel.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/cluster")
def cluster(req: ClusterRequest = Body(
        ...,
        example={
            "group" : 5,
            "place": [
    {"name": "가게1", "address": "서울특별시 종로구 세종대로 175", "latitude": 37.572950, "longitude": 126.976968},
    {"name": "가게2", "address": "서울특별시 종로구 율곡로 99", "latitude": 37.579617, "longitude": 126.991902},
    {"name": "가게3", "address": "서울특별시 중구 명동길 14", "latitude": 37.563679, "longitude": 126.982264},
    {"name": "가게4", "address": "서울특별시 중구 소공로 63", "latitude": 37.561002, "longitude": 126.983894},
    {"name": "가게5", "address": "서울특별시 용산구 이태원로 294", "latitude": 37.534700, "longitude": 126.994823},
    {"name": "가게6", "address": "서울특별시 용산구 한강대로 405", "latitude": 37.529998, "longitude": 126.967430},
    {"name": "가게7", "address": "서울특별시 강남구 테헤란로 521", "latitude": 37.507981, "longitude": 127.058118},
    {"name": "가게8", "address": "서울특별시 강남구 삼성로 512", "latitude": 37.513246, "longitude": 127.059529},
    {"name": "가게9", "address": "서울특별시 강남구 선릉로 627", "latitude": 37.502803, "longitude": 127.048630},
    {"name": "가게10", "address": "서울특별시 강서구 공항대로 248", "latitude": 37.561170, "longitude": 126.854173},
    {"name": "가게11", "address": "서울특별시 강서구 마곡중앙로 161", "latitude": 37.561664, "longitude": 126.825569},
    {"name": "가게12", "address": "서울특별시 은평구 진흥로 215", "latitude": 37.602970, "longitude": 126.929238},
    {"name": "가게13", "address": "서울특별시 은평구 통일로 680", "latitude": 37.609894, "longitude": 126.929507},
    {"name": "가게14", "address": "서울특별시 서초구 반포대로 222", "latitude": 37.500574, "longitude": 127.010581},
    {"name": "가게15", "address": "서울특별시 서초구 서초대로 396", "latitude": 37.493473, "longitude": 127.025512},
    {"name": "가게16", "address": "서울특별시 마포구 양화로 125", "latitude": 37.553686, "longitude": 126.918618},
    {"name": "가게17", "address": "서울특별시 마포구 독막로 320", "latitude": 37.548635, "longitude": 126.945088},
    {"name": "가게18", "address": "서울특별시 송파구 올림픽로 300", "latitude": 37.514219, "longitude": 127.104733},
    {"name": "가게19", "address": "서울특별시 송파구 송파대로 570", "latitude": 37.508678, "longitude": 127.087604},
    {"name": "가게20", "address": "서울특별시 관악구 남부순환로 1820", "latitude": 37.478161, "longitude": 126.951598},
    {"name": "가게21", "address": "서울특별시 관악구 봉천로 209", "latitude": 37.484528, "longitude": 126.941235},
    {"name": "가게22", "address": "서울특별시 동작구 상도로 369", "latitude": 37.502465, "longitude": 126.947947},
    {"name": "가게23", "address": "서울특별시 동작구 사당로 300", "latitude": 37.487565, "longitude": 126.981758},
    {"name": "가게24", "address": "서울특별시 성동구 왕십리로 222", "latitude": 37.561351, "longitude": 127.037810},
    {"name": "가게25", "address": "서울특별시 성동구 아차산로 103", "latitude": 37.547317, "longitude": 127.046485},
    {"name": "가게26", "address": "서울특별시 성북구 정릉로 77", "latitude": 37.602817, "longitude": 127.013111},
    {"name": "가게27", "address": "서울특별시 성북구 동소문로 287", "latitude": 37.592134, "longitude": 127.018494},
    {"name": "가게28", "address": "서울특별시 노원구 동일로 1415", "latitude": 37.654291, "longitude": 127.056162},
    {"name": "가게29", "address": "서울특별시 노원구 한글비석로 8", "latitude": 37.651266, "longitude": 127.061337},
    {"name": "가게30", "address": "서울특별시 도봉구 마들로 576", "latitude": 37.667491, "longitude": 127.042908}
  ]
        }
    )) -> Dict:
    try:
        coords = np.array([[p.latitude, p.longitude] for p in req.place], dtype=float)
        labels = run_kmeans(coords, k=req.group)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cluster_map: Dict[str, List[Place]] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(label, []).append(req.place[idx])
    
    sorted_result = {
        f"clustering{label + 1}": cluster_map[label]
        for label in sorted(cluster_map.keys())
    }

    return {"result": sorted_result}