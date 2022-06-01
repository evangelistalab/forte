from functools import wraps # This convenience func preserves name and docstring

def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self,*args, **kwargs): 
            return func(self,*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator
