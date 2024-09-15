

def delete_noisy_char(s):
    s = (
        s.replace(",", " ")
        .replace("/", " ")
        .replace('"', " ")
        .replace("-", " ")
        .replace(";", " ")
        .replace(".", " ")
        .replace("&", " ")
        .replace("?", " ")
        .replace("!", " ")
        .replace("(", " ")
        .replace(")", " ")
    )
    s = s.strip()
    return s


