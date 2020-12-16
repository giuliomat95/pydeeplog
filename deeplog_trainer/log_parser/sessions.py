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

    def split_sessions(self, anomaly_flag):
        for i in range(1, len(self.sessions)+1):
            if anomaly_flag[i] is False:
                self.normal_sessions[i] = self.sessions[i]
            else:
                self.abnormal_sessions[i] = self.sessions[i]
        return self.normal_sessions, self.abnormal_sessions
