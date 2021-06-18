class CallbackHandler():
    """
    This class stores a list of callback functions labeled by an ID.

    This class is used in solver classes to inject code at specific
    points in a computation, allowing the user to customize a computation
    without modifying code.
    The intended use for this function is the following. Add callback
    functions with specific IDs as::

        ch = CallbackHandler()
        def func(state):
            # do something with state
        # define callback with id='post' linked to the function func()
        ch.add_callback(id='post',func=func)

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
        self._callback_list = {}

    def add_callback(self, id, func):
        """Add a callback function labeled by an ID"""
        self._callback_list[id] = func

    def callback(self, id, state):
        """Call the function ID on a given state"""
        if id in self._callback_list:
            self._callback_list[id](state)