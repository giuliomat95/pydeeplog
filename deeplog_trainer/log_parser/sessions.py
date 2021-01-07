class SessionStorage:
    def __init__(self):
        self.sessions = {}
        self.templates = {}
        self.parameters = {}
        self.normal_sessions = {}
        self.abnormal_sessions = {}

    def get_sessions(self, sess_id, template_id):
        """
        It groups the log keys by session in a dictionary. Every 'key' corresponds a particular session Id, whose
        value is a collection of encoded log messages
        """
        if sess_id not in self.sessions.keys():
            self.sessions[sess_id] = [template_id]
        else:
            self.sessions[sess_id].append(template_id)
        return self.sessions

    def get_templates(self, template_id, template):
        """It links in a dictionary the templates Id with the corresponding text template"""
        self.templates[template_id] = template
        return self.templates

    def get_parameters(self, sess_id, parameter):
        """It return a dictionary with the list of parameters for each session Id"""
        if sess_id not in self.parameters.keys():
            self.parameters[sess_id] = [parameter]
        else:
            self.parameters[sess_id].append(parameter)
        return self.parameters

