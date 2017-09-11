from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base

from .resolvers import LOCAL_RESOLVERS
from .sql_types import NpArrayType

Base = declarative_base()


class ImageLocation(Base):
    __tablename__ = 'image_location'

    uid = Column(String(32), primary_key=True)
    resolver_key = Column(String, nullable=False)
    iiif_server_id = Column(String, nullable=False)

    def get_image_path(self):
        return LOCAL_RESOLVERS[self.resolver_key].resolves(self.iiif_server_id)


class RawFeature:
    uid = Column(String(32), primary_key=True)
    data = Column(NpArrayType)  # type: np.ndarray

    def __repr__(self):
        return "<{}(uid='{}', data_size={},{})>".format(self.__class__.__name__, self.uid, self.data.shape, self.data.dtype)


class Feature_Test(Base, RawFeature):
    __tablename__ = 'features'
    uid = Column(String(32), primary_key=True)
    data = Column(NpArrayType)  # type: np.ndarray


class QueryIterator:
    def __init__(self, query, fn=None, WINDOW_SIZE=1000):
        self.query = query
        self.WINDOW_SIZE = WINDOW_SIZE
        self.fn = fn if fn is not None else lambda x: x

    def __len__(self):
        return self.query.count()

    def __iter__(self):
        start = 0
        while True:
            stop = start + self.WINDOW_SIZE
            things = self.query.slice(start, stop).all()
            if len(things) == 0:
                break
            for thing in things:
                yield self.fn(thing)
            start += self.WINDOW_SIZE


# Adding an element :
# f = Feature(id=uid, data=np.zeros(2048))
# session.merge(f)

# Getting an element :
# session.query(Feature).filter_by(id=uid).one()