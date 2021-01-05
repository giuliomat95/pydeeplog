class SessionStorage:
    def __init__(self):
        self.sessions = {}
        self.templates = {}
        self.parameters = {}
        self.normal_sessions = {}
        self.abnormal_sessions = {}

    def get_sessions(self, sess_id, template_id):
        if sess_id not in self.sessions.keys():
            self.sessions[sess_id] = [template_id]
        else:
            self.sessions[sess_id].append(template_id)
        return self.sessions

    def get_templates(self, template_id, template):
        self.templates[template_id] = template
        return self.templates

    def get_parameters(self, sess_id, parameter):
        if sess_id not in self.parameters.keys():
            self.parameters[sess_id] = [parameter]
        else:
            self.parameters[sess_id].append(parameter)
        return self.parameters

