class BreakPoint:
    def __init__(self) -> None:
        pass
    
    def __call__(self):
        import ipdb; ipdb.set_trace() #FIXME