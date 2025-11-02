from dataclasses import dataclass as _dataclass, field, fields


def smart_dataclass(_cls=None, /, **kwargs):
    """
    A dataclass wrapper that:
    - Disables auto-generated __repr__
    - Injects a custom __repr__ that supports 'repr_name' metadata for aliasing field names
    """
    kwargs.setdefault("repr", False)

    def wrap(cls):
        # Apply dataclass transformation
        cls = _dataclass(**kwargs)(cls)

        # Define the unified __repr__ if not explicitly defined
        def __repr__(self):
            cls_name = type(self).__name__
            parts = []
            for f in fields(self):
                if not f.repr:
                    continue
                name = f.metadata.get("repr_name", f.name)
                val = getattr(self, f.name)
                parts.append(f"{name}={val!r}")
            return f"{cls_name}({', '.join(parts)})"

        # Attach it only if not already present
        if "__repr__" not in cls.__dict__:
            cls.__repr__ = __repr__

        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)


@smart_dataclass
class User:
    name: str
    _email: str = field(metadata={"repr_name": "email"})

    @property
    def email(self):
        return self._email


@smart_dataclass
class Admin(User):
    role: str = "admin"


u = User("Curtis", "cchong@example.com")
a = Admin("Alice", "alice@example.com", "superadmin")

print(u)
print(a)
