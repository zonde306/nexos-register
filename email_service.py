"""
邮箱服务类
"""
import asyncio
import os
import random
import re
import string

import requests
from dotenv import load_dotenv

load_dotenv()


class EmailService:
	"""邮箱服务类"""

	def __init__(self, timeout=30):
		"""初始化邮箱服务"""

		self.worker_domain = os.getenv("WORKER_DOMAIN")
		self.email_domain = os.getenv("EMAIL_DOMAIN")
		self.admin_password = os.getenv("ADMIN_PASSWORD")
		self.timeout = timeout
		self.jwt = None
		self.email = None

		if not all([self.worker_domain, self.email_domain, self.admin_password]):
			raise ValueError("Missing required environment variables: WORKER_DOMAIN, EMAIL_DOMAIN, ADMIN_PASSWORD")

	def _run_async(self, coro):
		"""在同步上下文中执行协程，兼容旧调用。"""
		try:
			asyncio.get_running_loop()
		except RuntimeError:
			return asyncio.run(coro)
		raise RuntimeError("检测到正在运行的事件循环，请改用 async 方法（*_async）")

	def _generate_random_name(self):
		"""生成随机邮箱名称"""
		letters1 = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 6)))
		numbers = ''.join(random.choices(string.digits, k=random.randint(1, 3)))
		letters2 = ''.join(random.choices(string.ascii_lowercase, k=random.randint(0, 5)))
		return letters1 + numbers + letters2

	async def create_email_async(self):
		"""异步创建临时邮箱。"""
		url = f"https://{self.worker_domain}/admin/new_address"
		try:
			random_name = self._generate_random_name()
			res = await asyncio.to_thread(
				requests.post,
				url,
				json={
					"enablePrefix": True,
					"name": random_name,
					"domain": self.email_domain,
				},
				headers={
					'x-admin-auth': self.admin_password,
					"Content-Type": "application/json"
				},
				timeout=self.timeout,
			)
			if res.status_code == 200:
				data = res.json()
				self.jwt = data.get('jwt')
				self.email = data.get('address')
				return self.jwt, self.email
			print(f"[-] 创建邮箱接口返回错误: {res.status_code} - {res.text}")
			return None, None
		except Exception as e:
			print(f"[-] 创建邮箱网络异常 ({url}): {e}")
			return None, None

	def create_email(self):
		"""兼容旧调用：同步创建临时邮箱。"""
		return self._run_async(self.create_email_async())

	async def create_temp_email_async(self):
		"""异步创建临时邮箱并返回邮箱地址。"""
		jwt, email = await self.create_email_async()
		if jwt and email:
			self.jwt = jwt
			self.email = email
			return email
		return None

	def create_temp_email(self):
		"""兼容旧调用：创建临时邮箱并返回邮箱地址"""
		return self._run_async(self.create_temp_email_async())

	async def get_emails_async(self):
		"""异步获取邮件列表。"""
		if not self.jwt:
			print("[-] 未创建邮箱")
			return []

		try:
			res = await asyncio.to_thread(
				requests.get,
				f"https://{self.worker_domain}/api/mails",
				params={"limit": 50, "offset": 0},
				headers={
					"Authorization": f"Bearer {self.jwt}",
					"Content-Type": "application/json"
				},
				timeout=self.timeout,
			)
			if res.status_code != 200:
				print(f"[-] 获取邮件接口返回错误: {res.status_code} - {res.text}")
				return []

			data = res.json() or {}
			results = data.get("results") or []
			emails = []
			for item in results:
				raw = item.get("raw") or item.get("text") or item.get("body") or ""
				subject = item.get("subject") or ""
				text_content = item.get("text_content") or item.get("text") or raw
				html_content = item.get("html_content") or item.get("html") or ""
				emails.append({
					"id": item.get("id") or item.get("mail_id") or item.get("createdAt") or raw[:32],
					"subject": subject,
					"text_content": text_content,
					"html_content": html_content,
					"raw": raw,
				})
			return emails
		except Exception as e:
			print(f"[-] 获取邮件网络异常: {e}")
			return []

	def get_emails(self):
		"""兼容旧调用：获取邮件列表"""
		return self._run_async(self.get_emails_async())

	async def get_verification_code_async(self, pattern=r'(\d{6})', timeout=60, poll_interval=5):
		"""异步轮询邮件并提取验证码。"""
		start = asyncio.get_running_loop().time()
		seen_ids = set()

		while asyncio.get_running_loop().time() - start < timeout:
			emails = await self.get_emails_async()
			for mail in emails:
				mail_id = mail.get("id")
				if mail_id in seen_ids:
					continue
				seen_ids.add(mail_id)

				content = f"{mail.get('subject', '')} {mail.get('text_content', '')} {mail.get('html_content', '')} {mail.get('raw', '')}"
				content = re.sub('<[^<]+?>', ' ', content)
				match = re.search(pattern, content, re.IGNORECASE)
				if match:
					return match.group(1) if match.groups() else match.group(0)
			await asyncio.sleep(poll_interval)

		return None

	def get_verification_code(self, pattern=r'(\d{6})', timeout=60):
		"""兼容旧调用：轮询邮件并提取验证码"""
		return self._run_async(self.get_verification_code_async(pattern=pattern, timeout=timeout))

	async def fetch_first_email_async(self, jwt, max_retries=3, retry_interval=1):
		"""异步获取首封邮件内容。"""
		last_error = None
		for i in range(max_retries):
			try:
				res = await asyncio.to_thread(
					requests.get,
					f"https://{self.worker_domain}/api/mails",
					params={"limit": 10, "offset": 0},
					headers={
						"Authorization": f"Bearer {jwt}",
						"Content-Type": "application/json"
					},
					timeout=self.timeout,
				)

				if res.status_code == 200:
					data = res.json() or {}
					results = data.get("results") or []
					if results:
						return results[0].get("raw")
					return None
				print(f"获取邮件失败: {res.text}")
				return None
			except Exception as e:
				print(f"获取邮件失败: {e}")
				last_error = e
				if i < max_retries - 1:
					await asyncio.sleep(retry_interval)

		raise last_error if last_error else RuntimeError("获取邮件失败: 未知错误")

	def fetch_first_email(self, jwt, max_retries=3):
		"""兼容旧调用：同步获取首封邮件内容。"""
		return self._run_async(self.fetch_first_email_async(jwt=jwt, max_retries=max_retries))
