class SessionStorageInterface:
    def add(self, msg):
        """Adds a message"""
        pass

    def get_all(self):
        """Returns all the stored sessions"""
        pass

class DimasSessionStorage(SessionStorageInterface):
    def __init__(self):
        self.sessions = []

    def add(self, msg: str):
        self.sessions.append(msg)

    def get_all(self) -> list:
        return self.sessions
