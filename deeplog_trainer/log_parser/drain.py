import re

class Drain:
    def __init__(self, template_miner):
        self.template_miner = template_miner

    def get_parameters(self, content, template):
        """
        Given the entire message and its template, it return a list of words masked in the template
        (what we call 'the variable part' of the message)
        """
        content = re.sub(r"\t+", r' ', content)
        content = re.sub(r" +", r' ', content)
        template_regex = re.sub(r"<.{1,5}>", "<*>", template)
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\\t+', r'\t', template_regex)
        template_regex = re.sub(r'\\ +', r' ', template_regex)  # replace any '\ ' by ' '
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, content)
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list

    def cluster_id_to_num(self, cluster_id):
        num = int(cluster_id[1:])
        return num

    def add_message(self, msg):
        """
        For each log message in input it return a dictionary with the correspondent template, template Id and list
        of parameters
        """
        msg = msg.strip()
        cluster = self.template_miner.add_log_message(msg)
        template = cluster['template_mined']
        template_id = cluster['cluster_id']
        template_id = self.cluster_id_to_num(template_id)
        parameter_list = self.get_parameters(msg, template)
        result = {
                'template_id': template_id,
                'template': template,
                'params': parameter_list
                }
        return result