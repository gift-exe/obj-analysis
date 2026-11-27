import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.orchestrator import _message_dict, answer_message
from app.api.serializers import sse
from app.core.db import get_db
from app.models import Conversation, Message, Video

router = APIRouter(prefix="/api")


@router.get("/conversations/{conversation_id}/messages")
def list_messages(conversation_id: uuid.UUID, db: Session = Depends(get_db)):
    if db.get(Conversation, conversation_id) is None:
        raise HTTPException(404, "conversation not found")
    msgs = db.scalars(select(Message).where(Message.conversation_id == conversation_id)
                      .order_by(Message.created_at))
    return [_message_dict(m) for m in msgs]


class NewMessage(BaseModel):
    content: str
    selection_id: str | None = None  # an object the user clicked on the video before asking


@router.post("/conversations/{conversation_id}/messages")
def post_message(conversation_id: uuid.UUID, body: NewMessage, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(404, "conversation not found")
    if db.get(Video, conv.video_id).status != "ready":
        raise HTTPException(409, "video is not ready")
    content = body.content.strip()
    if not content:
        raise HTTPException(400, "empty message")
    db.commit()  # don't hold this transaction (and its locks) open for the whole agent run

    def stream():
        for event, payload in answer_message(conversation_id, content, body.selection_id):
            yield sse(event, payload)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
