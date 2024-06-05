import uuid
from datetime import datetime
import json

def generate_unique_id():
    """Generate a unique session identifier."""
    return str(uuid.uuid4())

class SessionManager:
    """Manage session data for a machine learning training session."""
    def __init__(self):
        self.session_id = generate_unique_id()
        self.start_time = self.get_current_time()
        self.end_time = None
        self.data = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "model_details": {},
            "system_info": {},
            "training_data": {},
            "epochs": [],
            "completion_status": "",
            "final_metrics": {}
        }

    def get_current_time(self):
        """Return the current time in ISO 8601 format."""
        return datetime.now().isoformat()

    def close_session(self, status="Completed"):
        """Mark the session as completed or interrupted."""
        self.end_time = self.get_current_time()
        self.data["end_time"] = self.end_time
        self.data["completion_status"] = status

    def to_json(self):
        """Convert session data to a JSON format."""
        return json.dumps(self.data, indent=4)

# Example usage
if __name__ == "__main__":
    session = SessionManager()
    # Simulate adding some data
    session.close_session()
    print(session.to_json())
