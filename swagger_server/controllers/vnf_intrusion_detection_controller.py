import connexion
import six
#import json

from swagger_server.models.vnf_instance import VNFInstance  # noqa: E501
from swagger_server import util

import id_module as id

def get_vnf_info(prefix):
    
    sfc_vnfs = ["firewall", "dpi"]
    vnf_info = id.get_vnf_info(prefix, sfc_vnfs)
    vnfi_list = vnf_info["vnfi_list"]
    result = id.convert_vnf_info(vnfi_list)

    return result


def get_intrusion_detection_result(prefix):

    sfc_vnfs = ["firewall", "dpi"]
    result = id.get_intrusion_detection_result(prefix, sfc_vnfs)

    return result

def get_intrusion_detection_f1_score():
    return id.get_id_f1score()

def post_train_model():
    id.train_model()

def get_vnf_resource_usage(prefix):

    #sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    sfc_vnfs = ["firewall", "dpi"]
    result = id.get_vnf_resource_usage(prefix, sfc_vnfs)

    return result

def create_sfc(prefix):

    sfc_vnfs = ["firewall", "flowmonitor", "dpi", "ids", "lb"]
    result = id.get_vnf_resource_usage(prefix, sfc_vnfs)

    return result

