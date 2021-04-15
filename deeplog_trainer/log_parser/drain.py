import re
from importlib.metadata import version


class Drain:
    def __init__(self, template_miner):
        self.template_miner = template_miner

    def get_parameters(self, content, template):
        """
        Given the entire message and its template, it return a list of words
        masked in the template (what we call 'the variable part' of the message)
        """
        content = re.sub(r"\t+", r' ', content)
        content = re.sub(r" +", r' ', content)
        template_regex = re.sub(r"<.{1,5}>", "<*>", template)
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\\t+', r'\t', template_regex)
        # Replace any '\ ' by ' '
        template_regex = re.sub(r'\\ +', r' ', template_regex)
        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, content)
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if \
            isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list

    def cluster_id_to_num(self, cluster_id):
        """
        Extract the integer from the label composed by a letter in the first
        position followed by a 4 digits number.
        NB: the template codifications start from 1.
        """
        num = int(cluster_id[1:])
        return num

    def add_message(self, msg):
        """
        For each log message in input it returns a dictionary with the
        corresponding template, template Id and list of parameters
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

    def serialize_drain(self):
        clusters = [{'clusterId': cluster.cluster_id,
                     'logTemplateTokens': cluster.log_template_tokens}
                    for cluster in self.template_miner.drain.clusters]
        serialized = {'version': version('drain3'),
                      'depth': self.template_miner.drain.depth,
                      'similarityThreshold': self.template_miner.drain.sim_th,
                      'maxChildrenPerNode':
                          self.template_miner.drain.max_children,
                      'delimiters': self.template_miner.drain.extra_delimiters,
                      'clusters': clusters,
                      'root':
                          self._serialize_node(
                              self.template_miner.drain.root_node)
                      }
        return serialized

    def _serialize_node(self, node):
        tree_serialized = {
            'depth': node.depth,
            'key': node.key,
            'keyToChildNode': {child.key: self._serialize_node(child)
                               if len(node.key_to_child_node) > 0
                               else {}
                               for child in node.key_to_child_node.values()},
            'clusters': [{'clusterId': cluster.cluster_id,
                          'logTemplateTokens': cluster.log_template_tokens}
                         for cluster in node.clusters]
        }
        return tree_serialized
