from deeplog_trainer import SERIAL_DRAIN_VERSION


class Drain:
    def __init__(self, template_miner):
        self.template_miner = template_miner

    def add_message(self, msg):
        """
        For each log message in input it returns a dictionary with the
        corresponding template, template Id and list of parameters
        """
        msg = msg.strip()
        cluster = self.template_miner.add_log_message(msg)
        template = cluster['template_mined']
        template_id = cluster['cluster_id']
        parameter_list = self.template_miner.get_parameter_list(
            log_template=template, content=msg)
        result = {
            'template_id': template_id,
            'template': template,
            'params': parameter_list
        }
        return result

    def serialize_drain(self):
        masking = []
        for instruction in self.template_miner.config.masking_instructions:
            masking.append({'regex_pattern': instruction.regex_pattern,
                            'mask_with': '<' + instruction.mask_with + '>'})
        serialized = {'version': SERIAL_DRAIN_VERSION,
                      'depth': self.template_miner.drain.depth + 2,
                      'similarityThreshold': self.template_miner.drain.sim_th,
                      'maxChildrenPerNode':
                          self.template_miner.drain.max_children,
                      'delimiters':
                          [' ', *self.template_miner.drain.extra_delimiters],
                      'masking': masking,
                      'root':
                          self._serialize_node(
                              "root", self.template_miner.drain.root_node, 0)
                      }
        return serialized

    def _serialize_node(self, token, node, depth):
        tree_serialized = {
            'depth': depth,
            'key': token,
            'children': {
                token: self._serialize_node(token, child, depth + 1)
                if len(node.key_to_child_node) > 0
                else {}
                for token, child in node.key_to_child_node.items()
            },
            'clusters': [
                {'clusterId': cluster_id,
                 'logTemplateTokens':
                     list(self.template_miner.drain.id_to_cluster[
                         cluster_id].log_template_tokens)}
                for cluster_id in node.cluster_ids]
        }
        return tree_serialized
