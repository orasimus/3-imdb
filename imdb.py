import json

print(json.dumps({'step': 10, 'accuracy': 0.2}))
print(json.dumps({'step': 20, 'accuracy': 0.5}))
print(json.dumps({'step': 30, 'accuracy': 0.9}))
print(json.dumps({'step': 10, 'loss': 0.1}))
print(json.dumps({'step': 20, 'loss': 0.7}))
print(json.dumps({'step': 30, 'loss': 0.2}))
print(json.dumps({'step': 40, 'loss': 0.9}))
