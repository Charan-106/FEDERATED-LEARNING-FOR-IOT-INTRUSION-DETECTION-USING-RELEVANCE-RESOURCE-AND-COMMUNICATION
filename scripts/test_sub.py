import requests
r = requests.post('http://127.0.0.1:5000/api/auth/login', json={'email':'admin@arc-fl.local','password':'Admin@123'})
t = r.json()['token']
h = {'Authorization': f'Bearer {t}'}
r2 = requests.get('http://127.0.0.1:5000/api/admin/rounds', headers=h)
print('ROUNDS:', r2.json())
rounds = r2.json().get('rounds', [])
if rounds:
    r_id = rounds[0]['id']
    r3 = requests.get(f'http://127.0.0.1:5000/api/admin/rounds/{r_id}/submissions', headers=h)
    print('STATUS:', r3.status_code, r3.text)
