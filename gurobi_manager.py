import os

options = {
    "WLSACCESSID": os.environ.get("WLSACCESSID"),
    "WLSSECRET": os.environ.get("WLSSECRET"),
    "LICENSEID": int(os.environ.get("LICENSEID", 0)) or None,
}
