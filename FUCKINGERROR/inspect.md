AttributeError: module 'inspect' has no attribute 'getargspec'. Did you mean: 'getargs'?

```python
    want_out = 'out' in inspect.getargspec(func).args
                        ^^^^^^^^^^^^^^^^^^                     
```

https://stackoverflow.com/questions/74585622/pyfirmata-gives-error-module-inspect-has-no-attribute-getargspec

python 3.7(until 2023-06-27), 3.10(until 2026-10-04) 才行，3.11不行