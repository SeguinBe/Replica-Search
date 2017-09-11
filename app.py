import os

import requests
import sqlalchemy
from sqlalchemy.pool import StaticPool
from flask import Flask
from flask_restplus import Api, Resource
from flask_caching import Cache
from flask_prometheus import monitor
from sqlalchemy.orm import sessionmaker, scoped_session
from werkzeug.exceptions import BadRequest
import numpy as np

from replica_search import model, resolvers, index

app = Flask(__name__)
api = Api(app)

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

CACHE_SEARCH_TIMEOUT = 120  # Seconds

engine = sqlalchemy.create_engine('sqlite:////home/seguin/Replica-search/sqlalchemy.db',
                                  connect_args={'check_same_thread': False},
                                  poolclass=StaticPool)
model.Base.metadata.create_all(engine)
Session = scoped_session(sessionmaker(bind=engine))

DEFAULT_LOCAL_IMAGES_FOLDER = '/scratch/benoit/replica_local_downloaded_images'
DEFAULT_RESOLVER_KEY = 'default'
resolvers.LOCAL_RESOLVERS['iiif_replica'] = resolvers.LocalResolver('dhlabsrv4.epfl.ch/iiif_replica',
                                                                    '/mnt/homes/benoit/datasets/')
resolvers.LOCAL_RESOLVERS[DEFAULT_RESOLVER_KEY] = resolvers.DefaultResolver(DEFAULT_LOCAL_IMAGES_FOLDER)
RESIZING_MAX_DIM = 1024

for resolver in resolvers.LOCAL_RESOLVERS.values():
    assert os.path.exists(resolver.local_root_folder), resolver.local_root_folder

search_index = None


@api.route('/api/element/<string:uid>')
class RegisterImageResource(Resource):
    parser = api.parser()
    parser.add_argument('iiif_resource_url', type=str, required=True)
    parser.add_argument('overwrite', type=bool, default=False)

    @api.expect(parser)
    def post(self, uid):
        args = self.parser.parse_args()
        iiif_resource_url = args['iiif_resource_url']
        # Check that the resource is accessible and
        try:
            r = requests.get(iiif_resource_url)
            if r.status_code != 200:
                raise ValueError("{} is not accessible".format(iiif_resource_url))
            iiif_server_answer = r.json()
            # assert iiif_server_answer['protocol'].startswith('http://iiif.io/api/image')
        except Exception as e:
            raise BadRequest('Could not access or parse {} : {}'.format(iiif_resource_url, e))

        local_resolver_pair = resolvers.get_local_resolver_or_none(iiif_resource_url)
        if local_resolver_pair is not None:
            # Found a local version of the image
            resolver_key, local_path = local_resolver_pair
        else:
            # Needs to download the image to local storage
            resolver_key = DEFAULT_RESOLVER_KEY
            local_path = resolvers.generate_image_path(uid)
            output_path = os.path.join(DEFAULT_LOCAL_IMAGES_FOLDER, local_path)
            if args['overwrite'] or not os.path.exists(output_path):
                w, h = iiif_server_answer['width'], iiif_server_answer['height']
                sizes = iiif_server_answer.get('sizes')
                best_size_in_pyramid = None
                if sizes is not None:
                    for s in sizes:
                        if s[0] > RESIZING_MAX_DIM and s[1] > RESIZING_MAX_DIM and \
                                (best_size_in_pyramid is None or s[0] < best_size_in_pyramid[0]):
                            best_size_in_pyramid = s
                if best_size_in_pyramid is None:
                    if w < RESIZING_MAX_DIM and h < RESIZING_MAX_DIM:
                        desired_size = 'full'
                    elif w > h:
                        desired_size = '{},'.format(RESIZING_MAX_DIM)
                    else:
                        desired_size = ',{}'.format(RESIZING_MAX_DIM)
                else:
                    desired_size = '{},{}'.format(*best_size_in_pyramid)
                image_url = iiif_resource_url + '/full/{}/0/color.jpg'.format(desired_size)
                try:
                    print(image_url)
                    resolvers.download_resize_image(image_url, output_path, max_dim=RESIZING_MAX_DIM)
                except Exception as e:
                    raise BadRequest('Could not download an save image {} : {}'.format(image_url, str(e)))

        session = Session()
        img_loc = model.ImageLocation(uid=uid, iiif_server_id=local_path, resolver_key=resolver_key)
        session.merge(img_loc)
        session.commit()

    def get(self, uid):
        session = Session()


@api.route('/api/stats')
class GlobalStatistics(Resource):
    def get(self):
        session = Session()
        return {
            'nb_images_registered': session.query(sqlalchemy.func.count(model.ImageLocation.uid)).one(),
            'nb_images_per_resolver': {
                k: session.query(sqlalchemy.func.count(model.ImageLocation.uid)).filter(
                    model.ImageLocation.resolver_key == k).one()
                for k in resolvers.LOCAL_RESOLVERS.keys()
                }
        }


@api.route('/api/search')
class SearchResource(Resource):
    parser = api.parser()
    parser.add_argument('positive_image_uids', type=list, required=True, location='json')
    parser.add_argument('negative_image_uids', type=list, default=[], location='json')
    parser.add_argument('nb_results', type=int, default=100)

    @api.expect(parser)
    def post(self):
        args = self.parser.parse_args()
        if not algebraic_search_index:
            raise BadRequest('No index is loaded')

        @cache.memoize(timeout=CACHE_SEARCH_TIMEOUT, make_name='search')
        def _fn(args):
            results = algebraic_search_index.search(args['positive_image_uids'], args['negative_image_uids'], args['nb_results'])
            return {
                'results': [
                    {'uid': uid, 'score': s} for uid, s in results
                    ]
            }
        return _fn(args)


@api.route('/api/search_region')
class SearchResource(Resource):
    parser = api.parser()
    parser.add_argument('image_uid', type=str, required=True, location='json')
    parser.add_argument('box_x', type=float, required=True, location='json')
    parser.add_argument('box_y', type=float, required=True, location='json')
    parser.add_argument('box_h', type=float, required=True, location='json')
    parser.add_argument('box_w', type=float, required=True, location='json')
    parser.add_argument('nb_results', type=int, default=100)

    @api.expect(parser)
    def post(self):
        args = self.parser.parse_args()
        if not search_index:
            raise BadRequest('No index is loaded')

        @cache.memoize(timeout=CACHE_SEARCH_TIMEOUT, make_name='search_region')
        def _fn(args):
            results = search_index.search_region(args['image_uid'],
                                                 np.array([args['box_y'], args['box_x'], args['box_h'], args['box_w']]),
                                                 args['nb_results'])
            return {
                'results': [
                    {'uid': uid, 'score': s, 'box': {k: v for k, v in zip(['y', 'x', 'h', 'w'], box)}}
                    for uid, s, box in results
                    ]
            }

        return _fn(args)


@api.route('/api/distance_matrix')
class SearchResource(Resource):
    parser = api.parser()
    parser.add_argument('image_uids', type=list, required=True, location='json')

    @api.expect(parser)
    def post(self):
        args = self.parser.parse_args()
        if not search_index:
            raise BadRequest('No index is loaded')

        distance_matrix = algebraic_search_index.make_distance_matrix(args['image_uids'])

        return {'distances': distance_matrix.round(3).tolist()}


if __name__ == '__main__':
    algebraic_search_index = index.IntegralImagesIndex('/home/seguin/resnet_index.hdf5',
                                                       index.IntegralImagesIndex.IndexType.HALF_DIM_PCA)
    print('Initialized : {}'.format(algebraic_search_index))
    search_index = index.IntegralImagesIndex('/home/seguin/vgg_320_aug_triplet.hdf5')
    print('Initialized : {}'.format(search_index))
    monitor(app, port=5011)
    app.run(host='0.0.0.0', debug=True, threaded=True, port=5001)
