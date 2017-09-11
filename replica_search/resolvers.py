import os
from PIL import Image
import requests
from urllib.parse import urlparse, unquote
from typing import Dict, Union, Tuple


LOCAL_RESOLVERS = {}  # type: Dict[str, LocalResolver]


class LocalResolver:
    def __init__(self, base_url, local_root_folder):
        self.base_url = base_url
        self.local_root_folder = local_root_folder

    def matches(self, image_url) -> Union[None, str]:
        # Get rid of http:// or https://
        r = urlparse(image_url)
        base_url = r.netloc+r.path
        if base_url.startswith(self.base_url):
            iiif_resource_id = base_url[len(self.base_url):].strip('/')
            if not os.path.exists(self.resolves(iiif_resource_id)):
                raise ValueError('Resolver could not find {} at {}'.format(image_url,
                                                                           self.resolves(iiif_resource_id)))
            return iiif_resource_id
        else:
            return None

    def resolves(self, iiif_resource_id):
        return os.path.join(self.local_root_folder, unquote(iiif_resource_id))


class DefaultResolver(LocalResolver):
    def __init__(self, local_root_folder):
        super().__init__('', local_root_folder)

    def matches(self, image_url):
        return None


def generate_image_path(uid: str) -> str:
    return os.path.join(uid[0], uid[1], uid[2:] + '.jpg')


def download_resize_image(image_url: str, output_path: str, max_dim=1024, timeout=10) -> None:
    r = requests.get(image_url, stream=True, headers={'User-agent': 'Mozilla/5.0'}, timeout=timeout)
    if r.status_code != 200:
        raise ValueError("{} is not accessible".format(image_url))
    r.raw.decode_content = True
    image = Image.open(r.raw).convert('RGB')  # type: Image
    w, h = image.size
    new_w, new_h = int(w*max_dim/max(w, h)), int(h*max_dim/max(w, h))
    image = image.resize((new_w, new_h), resample=Image.BILINEAR)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def get_local_resolver_or_none(image_url: str) -> Union[None, Tuple[str, str]]:
    for k, resolver in LOCAL_RESOLVERS.items():
        iiif_resource_id = resolver.matches(image_url)
        if iiif_resource_id is not None:
            return k, iiif_resource_id
    return None


