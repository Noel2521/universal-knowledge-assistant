from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# In-memory conversation store
# Key: conversation_id, Value: list of messages
conversation_store: Dict[str, List[Dict]] = {}

def get_conversation_history(conversation_id: str) -> List[Dict]:
    """
    Get conversation history for a given conversation ID.
    """
    history = conversation_store.get(conversation_id, [])
    logger.info(f"Retrieved {len(history)} messages for {conversation_id}")
    return history

def add_to_conversation(
    conversation_id: str,
    question: str,
    answer: str
) -> None:
    """
    Add a question and answer to conversation history.
    """
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = []

    conversation_store[conversation_id].append(
        {"role": "user", "content": question}
    )
    conversation_store[conversation_id].append(
        {"role": "assistant", "content": answer}
    )
    logger.info(f"Added exchange to conversation {conversation_id}")

def clear_conversation(conversation_id: str) -> None:
    """
    Clear conversation history for a given conversation ID.
    """
    if conversation_id in conversation_store:
        conversation_store.pop(conversation_id)
        logger.info(f"Cleared conversation {conversation_id}")
    else:
        logger.warning(f"Conversation {conversation_id} not found")