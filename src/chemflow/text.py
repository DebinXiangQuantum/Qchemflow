import hashlib
import json

from chemflow.diagnostics import TextReferenceError


def cap_strings(value: object, limit: int = 2048, head: int = 200) -> object:
    if isinstance(value, str) and len(value) > limit:
        return {"text_chars": len(value), "text_sha256": hashlib.sha256(value.encode()).hexdigest(),
                "head": value[:head],
                "ref": "pass this object back as the parameter value to reuse the frozen text"}
    if isinstance(value, dict):
        return {key: cap_strings(item, limit, head) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cap_strings(item, limit, head) for item in value]
    return value


class FrozenTextReferences:
    def __init__(self):
        self._texts: dict[str, tuple[str, str]] = {}

    def collect(self, value: object, where: str) -> None:
        if isinstance(value, str) and len(value) > 2048:
            digest = hashlib.sha256(value.encode()).hexdigest()
            self._texts.setdefault(digest, (value, where))
        elif isinstance(value, dict):
            for key, item in value.items():
                self.collect(item, f"{where}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                self.collect(item, f"{where}[{index}]")

    def resolve(self, value: object) -> object:
        if isinstance(value, dict):
            if "text_sha256" in value:
                digest = value["text_sha256"]
                match = self._texts.get(digest) if isinstance(digest, str) else None
                if match is None or ("text_chars" in value and
                        (type(value["text_chars"]) is not int or
                         value["text_chars"] != len(match[0]))):
                    available = [{"text_sha256": digest[:12], "text_chars": len(text),
                                  "where": where}
                                 for digest, (text, where) in list(self._texts.items())[:3]]
                    raise TextReferenceError(json.dumps(available))
                return match[0]
            return {key: self.resolve(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.resolve(item) for item in value]
        return value
