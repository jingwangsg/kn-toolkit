#https://www.zhihu.com/question/269707221/answer/2677167861
def commit(content):
    import git
    repo = git.Repo(search_parent_directories=True)
    try:
        g = repo.git
        g.add("--all")
        res = g.commit("-m " + content)
        print(res)
    except Exception as e:
        print("no need to commit")

def get_origin_url():
    import git
    repo = git.Repo(search_parent_directories=True)
    remote = repo.remote()
    remote_url = remote.url
    return remote_url