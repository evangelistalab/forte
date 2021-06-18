class Callback():
    """
    This class stores a callback functions, its ID, and any reported value.

    The function `func` must have signature `func(cb,state)`, where cb will
    be this Callback object and state is the state on which we want the
    callback function to act upon.
    """
    def __init__(self, id, func):
        self._id = id
        self._func = func
        self._report = None

    @property
    def report(self):
        return self._report

    @report.setter
    def report(self, val):
        self._report = val


class CallbackHandler():
    """
    This class stores a dictionary of callback functions labeled by an ID.

    This class is used in solver classes to inject code at specific
    points in a computation, allowing the user to customize a computation
    without modifying code.
    The intended use for this function is the following. Add callback
    functions with specific IDs as::

        cbh = CallbackHandler()
        def func(cb,state):
            # do something with state
        # define callback with id='post' linked to the function func()
        cbh.add_callback(id='post',func=func)

    This object can now be passed to a class that takes a CallbackHandler
    and that calls::

        class SomeMethod():
            '''A class that implements callbacks'''
            def __init__(self,...,cb = None):
                '''Store the callback object'''
                self.cb = cb

            def run(self,...):
                '''Run a computation calling the callback object at several points'''
                self.ch.callback('pre',self) # nothing happens here
                # some computation here
                self.ch.callback('post',self) # calls func(self)

    When the method ``run()`` is executed, the second callback will be triggered
    calling ``func(self)``.
    """
    def __init__(self):
        self._callback_dict = {}

    def add_callback(self, id, func):
        """Add a callback function labeled by an ID"""
        self._callback_dict[id] = Callback(id, func)

    def call(self, id, state):
        """Call the function ID on a given state"""
        if id in self._callback_dict:
            self._callback_dict[id]._func(self._callback_dict[id], state)

    def report(self, id):
        """Return a report for function ID"""
        if id in self._callback_dict:
            return self._callback_dict[id].report
        return None
