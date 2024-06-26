from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from base64 import b64encode, b64decode
#created by 733n_wolf on Sun Apr 28 2024 05:01:54 GMT+0530 (India Standard Time)
key = get_random_bytes(32)
cipher = AES.new(key, AES.MODE_ECB)
flag = "3VQa2xUIX45Q94oQGPznGYtNz9bGge2CZeWx6ITACVI="
padded_flag = flag.encode().rjust(len(flag) + (16 - len(flag) % 16))
ciphertext = cipher.encrypt(padded_flag)
encoded_ciphertext = b64encode(ciphertext).decode()
print(encoded_ciphertext)