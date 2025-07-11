from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    path: str


class CharacterInfo(BaseModel):
    name: str
    gender: str
    age: str
    job: str
    relationship: str

    class Config:
        extra = "ignore"


class UserFNames(BaseModel):
    chat_json_fname: str
    chat_txt_fname: str
    report_txt_fname: str


class DailyReport(BaseModel):
    overall_score: int
    mental_health_score: int
    interests_needs_score: int
    family_connection_score: int
    safety_awareness_score: int
    physical_health_score: int
    living_conditions_score: int
    overall_comments: str
    mental_health_comments: str
    interests_needs_comments: str
    family_connection_comments: str
    safety_awareness_comments: str
    physical_health_comments: str
    living_conditions_comments: str