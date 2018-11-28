import os

import requests
import sqlalchemy
from sqlalchemy.pool import StaticPool
from flask import Flask, send_file
from flask_restplus import Api, Resource
from flask_caching import Cache
from flask_prometheus import monitor
from sqlalchemy.orm import sessionmaker, scoped_session
from werkzeug.exceptions import BadRequest
import numpy as np

from replica_search import model, resolvers, index, duplicates
import io
import base64

app = Flask(__name__)
api = Api(app)

app.config.from_object('config')

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

CACHE_SEARCH_TIMEOUT = 120  # Seconds

engine = sqlalchemy.create_engine('sqlite:///' + app.config['SQLITE_FILE'],
                                  connect_args={'check_same_thread': False},
                                  poolclass=StaticPool)
model.Base.metadata.create_all(engine)
Session = scoped_session(sessionmaker(bind=engine))

# where to store downloaded added images, if None, is disabled
DEFAULT_LOCAL_IMAGES_FOLDER = app.config['DEFAULT_LOCAL_IMAGES_FOLDER']
DEFAULT_RESOLVER_KEY = 'default'

for resolver_key, base_url, local_root_folder in app.config['LOCAL_RESOLVERS']:
    resolvers.LOCAL_RESOLVERS[resolver_key] = resolvers.LocalResolver(base_url, local_root_folder)
resolvers.LOCAL_RESOLVERS[DEFAULT_RESOLVER_KEY] = resolvers.DefaultResolver(DEFAULT_LOCAL_IMAGES_FOLDER)
RESIZING_MAX_DIM = 1024

for resolver in resolvers.LOCAL_RESOLVERS.values():
    assert os.path.exists(resolver.local_root_folder), resolver.local_root_folder

search_indexes = dict()
DEFAULT_SEARCH_INDEX_KEY = 'default'
DEFAULT_REGION_SEARCH_INDEX_KEY = DEFAULT_SEARCH_INDEX_KEY
DEFAULT_ALGEBRAIC_SEARCH_INDEX_KEY = 'algebraic'


def get_search_index(key, default_key=DEFAULT_SEARCH_INDEX_KEY) -> index.IntegralImagesIndex:
    correct_key = key if key is not None else default_key
    search_index = search_indexes[correct_key]
    if not search_index:
        raise BadRequest('Index not found for key {}'.format(correct_key))
    return search_index


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
            assert DEFAULT_LOCAL_IMAGES_FOLDER is not None, "Local storage capabilities not activated"
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


@api.route('/api/transition_gif/<string:uid1>/<string:uid2>')
class TransitionGifResource(Resource):
    def get(self, uid1, uid2):
        session = Session()
        img_loc1 = session.query(model.ImageLocation).filter(model.ImageLocation.uid == uid1).one()
        img_loc2 = session.query(model.ImageLocation).filter(model.ImageLocation.uid == uid2).one()
        img_path1 = img_loc1.get_image_path()
        img_path2 = img_loc2.get_image_path()
        file_handle = io.BytesIO()
        duplicates.make_transition_gif(img_path1, img_path2, file_handle)
        file_handle.seek(0)
        return send_file(
            file_handle,
            attachment_filename='transition.gif',
            mimetype='image/gif'
        )


@api.route('/api/transition_gif_validity/<string:uid1>/<string:uid2>')
class TransitionGifResource(Resource):
    def get(self, uid1, uid2):
        session = Session()
        img_loc1 = session.query(model.ImageLocation).filter(model.ImageLocation.uid == uid1).one()
        img_loc2 = session.query(model.ImageLocation).filter(model.ImageLocation.uid == uid2).one()
        img_path1 = img_loc1.get_image_path()
        img_path2 = img_loc2.get_image_path()
        is_valid = duplicates.is_transition_valid(img_path1, img_path2)
        return {"valid": is_valid}


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
    parser.add_argument('positive_image_uids', type=list, default=[], location='json')
    parser.add_argument('negative_image_uids', type=list, default=[], location='json')
    parser.add_argument('positive_images_b64', type=list, default=[], location='json')
    parser.add_argument('negative_images_b64', type=list, default=[], location='json')
    parser.add_argument('nb_results', type=int, default=100)
    parser.add_argument('index', type=str, location='json')
    parser.add_argument('filtered_uids', type=list, default=[], location='json')
    parser.add_argument('rerank', type=bool, default=False, location='json')

    @api.expect(parser)
    def post(self):
        args = self.parser.parse_args()

        # print(args)

        @cache.memoize(timeout=CACHE_SEARCH_TIMEOUT, make_name='search')
        def _fn(args):
            search_index = get_search_index(args['index'], DEFAULT_ALGEBRAIC_SEARCH_INDEX_KEY)

            if len(args['positive_image_uids']) + len(args['positive_images_b64']) == 0:
                return {'results': [], 'total': search_index.get_number_of_images()}

            positive_features = np.stack(
                [search_index.get_feature_from_uuid(uid) for uid in args['positive_image_uids']] +
                [search_index.get_feature_from_image(base64.urlsafe_b64decode(img_b64.encode()))
                 for img_b64 in args['positive_images_b64']]
            )

            if len(args['negative_image_uids']) + len(args['negative_images_b64']) > 0:
                negative_features = np.stack(
                    [search_index.get_feature_from_uuid(uid) for uid in args['negative_image_uids']] +
                    [search_index.get_feature_from_image(base64.urlsafe_b64decode(img_b64.encode()))
                     for img_b64 in args['negative_images_b64']]
                )
            else:
                negative_features = np.zeros((0, positive_features.shape[1]), np.float32)

            if len(positive_features) == 1 and len(args['positive_image_uids']) == 1 and len(args['negative_image_uids']) == 0 and args['rerank']:
                # results = search_index.search(args['positive_image_uids'],
                #                              args['negative_image_uids'],
                #                              max(1000, args['nb_results']),
                #                              args['filtered_uids'])
                results = search_index.search_from_features(positive_features, negative_features,
                                                            max(1000, args['nb_results']),
                                                            args['filtered_uids'])
                integral_index = get_search_index(args['index'], DEFAULT_REGION_SEARCH_INDEX_KEY)
                results = integral_index.search_with_cnn_reranking(args['positive_image_uids'][0],
                                                                   args['nb_results'],
                                                                   filtered_ids=args['filtered_uids'],
                                                                   candidates=[r[0] for r in results])
                return {
                    'results': [
                        {'uid': uid, 'score': s, 'box': {k: v for k, v in zip(['y', 'x', 'h', 'w'], box)}}
                        for uid, s, box in results
                        ],
                    'total': len(args['filtered_uids']) if args['filtered_uids']
                    else integral_index.get_number_of_images()
                }
            else:
                #results = search_index.search(args['positive_image_uids'], args['negative_image_uids'],
                #                              args['nb_results'], args['filtered_uids'])
                results = search_index.search_from_features(positive_features, negative_features,
                                                            args['nb_results'],
                                                            args['filtered_uids'])
                return {
                    'results': [
                        {'uid': uid, 'score': s} for uid, s in results
                        ],
                    'total': len(args['filtered_uids']) if args['filtered_uids']
                    else search_index.get_number_of_images()
                }

        return _fn(args)


@api.route('/api/search_external')
class SearchResource(Resource):
    parser = api.parser()
    parser.add_argument('image_b64', type=str, required=True, location='json')
    parser.add_argument('nb_results', type=int, default=100)
    parser.add_argument('filtered_uids', type=list, default=[], location='json')
    parser.add_argument('rerank', type=bool, default=False, location='json')

    @api.expect(parser)
    def post(self):
        args = self.parser.parse_args()

        def _fn(args):
            search_index = get_search_index(DEFAULT_ALGEBRAIC_SEARCH_INDEX_KEY)
            # Decompress image
            image_b64 = args['image_b64'].encode()  # type: bytes
            image_bytes = base64.urlsafe_b64decode(image_b64)
            # Search
            results = search_index.search_from_image(image_bytes, args['nb_results'], args['filtered_uids'])
            return {
                'results': [
                    {'uid': uid, 'score': s} for uid, s in results
                    ],
                'total': len(args['filtered_uids']) if args['filtered_uids']
                else search_index.get_number_of_images()
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
    parser.add_argument('index', type=str, default=DEFAULT_REGION_SEARCH_INDEX_KEY, location='json')
    parser.add_argument('filtered_uids', type=list, default=[], location='json')

    @api.expect(parser)
    def post(self):
        args = self.parser.parse_args()

        @cache.memoize(timeout=CACHE_SEARCH_TIMEOUT, make_name='search_region')
        def _fn(args):
            search_index = get_search_index(args['index'], DEFAULT_REGION_SEARCH_INDEX_KEY)
            results = search_index.search_region(args['image_uid'],
                                                 np.array([args['box_y'], args['box_x'], args['box_h'], args['box_w']]),
                                                 args['nb_results'],
                                                 filtered_ids=args['filtered_uids'])
            return {
                'results': [
                    {'uid': uid, 'score': s, 'box': {k: v for k, v in zip(['y', 'x', 'h', 'w'], box)}}
                    for uid, s, box in results
                    ],
                'total': len(args['filtered_uids']) if args['filtered_uids']
                else search_index.get_number_of_images()
            }

        return _fn(args)


@api.route('/api/distance_matrix')
class SearchResource(Resource):
    parser = api.parser()
    parser.add_argument('image_uids', type=list, required=True, location='json')
    parser.add_argument('index', type=str, location='json')

    @api.expect(parser)
    def post(self):
        args = self.parser.parse_args()
        # print(args)
        search_index = get_search_index(args['index'], DEFAULT_ALGEBRAIC_SEARCH_INDEX_KEY)

        distance_matrix = search_index.make_distance_matrix(args['image_uids'])

        return {'distances': distance_matrix.round(3).tolist()}


if __name__ == '__main__':
    for search_index_key, filename, index_key in app.config['SEARCH_INDEXES']:
        search_indexes[search_index_key] = index.IntegralImagesIndex(filename, index_key)

    # search_indexes['untrained'] = index.IntegralImagesIndex('/home/seguin/resnet_index_untrained.hdf5')
    # for exp_name in ['canaletto_guardi', 'tiziano', 'allegory', 'diana']:
    #    search_indexes[exp_name] = index.IntegralImagesIndex('/home/seguin/experiment_{}_index.hdf5'.format(exp_name))
    for k, v in search_indexes.items():
        print('Initialized "{}" : {}'.format(k, v))
    # monitor(app, port=5011)
    app.run(host='0.0.0.0', debug=False, threaded=True, port=5001)
