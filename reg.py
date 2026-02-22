"""
Workspace 自动化脚本

"""

import quopri
from playwright.async_api import Page
from camoufox.async_api import AsyncCamoufox
import asyncio
import os
import time
import random
import string
import cv2
import contextlib
from pathlib import Path
# from ext import mail_service  # 旧的邮箱服务（已注释）
from email_service import EmailService
import logging

logger = logging.getLogger(__name__)

# ========== 代理配置 ==========
# 使用本地 v2ray 代理
USE_PROXY = os.getenv("USE_PROXY", "0") == "1"
LOCAL_PROXY = os.getenv("LOCAL_PROXY", "http://127.0.0.1:11080")  # v2ray 本地 HTTP 代理地址

# ========== Email 服务配置 ==========
EMAIL_SERVICE_TIMEOUT = 15


# 创建邮箱服务实例
mail = EmailService(timeout=EMAIL_SERVICE_TIMEOUT)


def generate_password(length=10):
    """生成符合要求的随机密码：大写+小写+数字+特殊字符"""
    password = [
        random.choice(string.ascii_uppercase),  # 至少1个大写
        random.choice(string.ascii_lowercase),  # 至少1个小写
        random.choice(string.digits),           # 至少1个数字
        random.choice('!@#$%^&*'),              # 至少1个特殊字符
    ]
    # 填充剩余长度
    remaining = length - 4
    password.extend(random.choices(
        string.ascii_letters + string.digits + '!@#$%^&*', 
        k=remaining
    ))
    random.shuffle(password)
    return ''.join(password)


async def human_like_mouse_move(page: Page, target_x: float, target_y: float, duration: float = 0.5):
    """
    模拟人类的鼠标移动轨迹（贝塞尔曲线）
    
    参数:
        page: Playwright Page 对象
        target_x: 目标 X 坐标
        target_y: 目标 Y 坐标
        duration: 移动持续时间（秒）
    """
    # 获取当前鼠标位置（假设从随机起点开始）
    start_x = random.uniform(100, 500)
    start_y = random.uniform(100, 400)
    
    # 生成贝塞尔曲线控制点
    control_x1 = start_x + random.uniform(-100, 100)
    control_y1 = start_y + random.uniform(-100, 100)
    control_x2 = target_x + random.uniform(-50, 50)
    control_y2 = target_y + random.uniform(-50, 50)
    
    # 计算移动步数
    steps = int(duration * 100)  # 每秒 100 步
    
    for i in range(steps + 1):
        t = i / steps
        
        # 三次贝塞尔曲线公式
        x = (1-t)**3 * start_x + \
            3 * (1-t)**2 * t * control_x1 + \
            3 * (1-t) * t**2 * control_x2 + \
            t**3 * target_x
        
        y = (1-t)**3 * start_y + \
            3 * (1-t)**2 * t * control_y1 + \
            3 * (1-t) * t**2 * control_y2 + \
            t**3 * target_y
        
        # 添加微小的随机抖动
        x += random.uniform(-1, 1)
        y += random.uniform(-1, 1)
        
        await page.mouse.move(x, y)
        await asyncio.sleep(duration / steps)


def find_checkbox_in_screenshot(screenshot_path: str, debug=False) -> tuple:
    """
    使用图像匹配优先定位 Turnstile，然后回退到 OCR/边缘检测

    参数:
        screenshot_path: 截图文件路径
        debug: 是否保存调试图片

    返回:
        tuple: (x, y, found) - 复选框中心坐标和是否找到
    """
    try:
        # 读取截图
        img = cv2.imread(screenshot_path)
        if img is None:
            return (0, 0, False)

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ========== 方法1：模板匹配（目标模板：171035.png） ==========
        # 171035.png 为 Turnstile 条形区域（约 380x65），复选框中心相对偏移约 (30, 32)
        template_paths = [
            Path("171035.png"),
            Path(__file__).resolve().parent / "171035.png",
        ]
        template_path = next((p for p in template_paths if p.exists()), None)

        if template_path is not None:
            template_img = cv2.imread(str(template_path))
            if template_img is not None:
                template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

                best_score = -1.0
                best_loc = None
                best_size = None

                # 适配不同缩放/DPI
                for scale in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]:
                    resized_template = cv2.resize(
                        template_gray,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
                    )
                    th, tw = resized_template.shape[:2]

                    # 过滤无效尺寸
                    if tw < 40 or th < 20:
                        continue
                    if tw > gray.shape[1] or th > gray.shape[0]:
                        continue

                    result = cv2.matchTemplate(gray, resized_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)

                    if max_val > best_score:
                        best_score = max_val
                        best_loc = max_loc
                        best_size = (tw, th)

                # 阈值可根据现场截图微调；0.45 对轻微缩放/压缩更稳健
                if best_loc is not None and best_size is not None and best_score >= 0.45:
                    checkbox_x = int(best_loc[0] + best_size[0] * (30 / 380))
                    checkbox_y = int(best_loc[1] + best_size[1] * (32 / 65))

                    print(
                        f"Template matched ({template_path.name}) score={best_score:.3f}, "
                        f"checkbox at ({checkbox_x}, {checkbox_y})"
                    )

                    if debug:
                        debug_img = img.copy()
                        x1, y1 = best_loc
                        x2, y2 = x1 + best_size[0], y1 + best_size[1]
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 165, 0), 2)
                        cv2.circle(debug_img, (checkbox_x, checkbox_y), 15, (0, 0, 255), 2)
                        cv2.circle(debug_img, (checkbox_x, checkbox_y), 3, (255, 0, 0), -1)

                        debug_path = Path(screenshot_path).parent / f"debug_template_{Path(screenshot_path).name}"
                        cv2.imwrite(str(debug_path), debug_img)

                    return (checkbox_x, checkbox_y, True)
                else:
                    print(f"Template matching score too low: {best_score:.3f}")
            else:
                print(f"Failed to read template image: {template_path}")
        else:
            print("Template file 171035.png not found, fallback to OCR")

        # ========== 方法2：OCR（回退） ==========
        import pytesseract

        # 配置 tesseract 路径（Windows）
        if os.name == 'nt':  # Windows
            tesseract_paths = [
                r'D:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                os.getenv('TESSERACT_PATH', "./tesseract.exe")
            ]
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break

        # 使用 OCR 识别文字和位置（简体+繁体中文）
        custom_config = r'--oem 3 --psm 6 -l chi_sim+chi_tra'

        try:
            ocr_data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"OCR failed with chi_tra, trying chi_sim only: {e}")
            # 如果繁体中文包不存在，只用简体
            custom_config = r'--oem 3 --psm 6 -l chi_sim'
            ocr_data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # 打印所有识别到的文字（调试用）
        n_boxes = len(ocr_data['text'])
        all_texts = []
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if text:
                all_texts.append(text)
        
        if all_texts:
            print(f"OCR found texts: {' '.join(all_texts[:20])}")  # 只显示前20个词
        
        # 查找"确认"或"確認"文字
        text_candidates = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            if not text:
                continue
            
            # 只查找包含"确认"或"確認"的文字
            if '确认' in text or '確認' in text:
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                
                text_candidates.append({
                    'x': x, 'y': y, 'w': w, 'h': h, 
                    'text': text
                })
        
        # 如果没找到"确认"，尝试查找"您"、"真人"等关键字
        if not text_candidates:
            logger.debug("'确认' not found, trying to find '您' or '真人'")
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                if not text:
                    continue
                
                # 查找"您"、"真"、"人"等字
                if any(char in text for char in ['您', '真', '人', '確', '認']):
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    text_candidates.append({
                        'x': x, 'y': y, 'w': w, 'h': h, 
                        'text': text
                    })
        
        if not text_candidates:
            print("No Chinese text found, trying edge detection")
            print(all_texts)
            return find_checkbox_by_edges(screenshot_path, img, gray, debug)
        
        # 选择 y 坐标最大的（最靠下的），这通常是 Turnstile 的文字
        text_candidates.sort(key=lambda t: t['y'], reverse=True)
        text_position = text_candidates[0]
        
        # 计算复选框位置：在文字左边 28 像素
        # 如果识别到的是"您"或其他字（不是"确认"），需要调整偏移量
        text_center_y = text_position['y'] + text_position['h'] // 2
        
        if '确认' in text_position['text'] or '確認' in text_position['text']:
            # 识别到"确认"，使用标准偏移
            checkbox_x = text_position['x'] - 28
            logger.info(f"Found '确认' at ({text_position['x']}, {text_position['y']}), checkbox at ({checkbox_x}, {text_center_y})")
        else:
            # 识别到其他字（如"您"、"真人"），这些字在"确认"右边，需要更大的偏移
            checkbox_x = text_position['x'] - 60
            logger.info(f"Found '{text_position['text']}' at ({text_position['x']}, {text_position['y']}), checkbox at ({checkbox_x}, {text_center_y})")
        
        checkbox_y = text_center_y
        
        # 如果需要调试，保存标注图片
        if debug:
            debug_img = img.copy()
            # 画出文字位置（绿色）
            cv2.rectangle(debug_img,
                        (text_position['x'], text_position['y']),
                        (text_position['x'] + text_position['w'], text_position['y'] + text_position['h']),
                        (0, 255, 0), 2)
            
            # 画出计算的复选框位置（红色）
            cv2.circle(debug_img, (checkbox_x, checkbox_y), 15, (0, 0, 255), 2)
            cv2.circle(debug_img, (checkbox_x, checkbox_y), 3, (255, 0, 0), -1)
            
            debug_path = Path(screenshot_path).parent / f"debug_{Path(screenshot_path).name}"
            cv2.imwrite(str(debug_path), debug_img)
        
        return (checkbox_x, checkbox_y, True)
        
    except ImportError:
        print("pytesseract not available, using edge detection")
        img = cv2.imread(screenshot_path)
        if img is None:
            return (0, 0, False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return find_checkbox_by_edges(screenshot_path, img, gray, debug)
    except Exception as e:
        print(f"OCR error: {e}")
        img = cv2.imread(screenshot_path)
        if img is None:
            return (0, 0, False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return find_checkbox_by_edges(screenshot_path, img, gray, debug)


def find_checkbox_by_edges(screenshot_path: str, img, gray, debug=False) -> tuple:
    """
    回退方法：使用边缘检测查找复选框（当 OCR 失败时使用）
    
    参数:
        screenshot_path: 截图文件路径
        img: 原始图片
        gray: 灰度图
        debug: 是否保存调试图片
    
    返回:
        tuple: (x, y, found) - 复选框中心坐标和是否找到
    """
    try:
        img_height, img_width = img.shape[:2]
        
        # 使用边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        logger.debug(f"Found {len(contours)} contours")
        
        # 查找方形轮廓（复选框通常是正方形）
        checkbox_candidates = []
        
        for contour in contours:
            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 复选框特征：
            # 1. 宽高接近（正方形）
            # 2. 大小在 15-30 像素之间
            # 3. 宽高比接近 1
            # 4. 位置：Turnstile 通常在页面中间偏左，垂直位置在 30%-60% 之间
            aspect_ratio = w / h if h > 0 else 0
            
            if 15 <= w <= 30 and 15 <= h <= 30 and 0.8 <= aspect_ratio <= 1.2:
                # 计算中心点
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Turnstile 复选框的位置特征：
                # - 水平：通常在左侧 10%-25% 的位置
                # - 垂直：通常在 35%-65% 的位置（中间）
                x_ratio = center_x / img_width
                y_ratio = center_y / img_height
                
                if 0.10 < x_ratio < 0.25 and 0.35 < y_ratio < 0.65:
                    checkbox_candidates.append({
                        'x': center_x,
                        'y': center_y,
                        'w': w,
                        'h': h,
                        'area': w * h,
                        'aspect_ratio': aspect_ratio,
                        'x_ratio': x_ratio,
                        'y_ratio': y_ratio
                    })
        
        if not checkbox_candidates:
            logger.debug("No checkbox candidates found in screenshot")
            return (0, 0, False)
        
        logger.debug(f"Found {len(checkbox_candidates)} checkbox candidates")
        
        # 选择最可能的复选框
        # 优先选择：1) 面积接近 20x20 2) 宽高比接近1 3) 位置在左侧中间
        for candidate in checkbox_candidates:
            candidate['score'] = (
                abs(candidate['area'] - 400) * 0.5 +  # 20x20 是理想大小
                abs(candidate['aspect_ratio'] - 1.0) * 200 +  # 越接近正方形越好
                abs(candidate['x_ratio'] - 0.16) * 500 +  # 期望在 16% 的位置
                abs(candidate['y_ratio'] - 0.59) * 300  # 期望在 59% 的位置（根据实际位置 427/720）
            )
        
        checkbox_candidates.sort(key=lambda c: c['score'])
        best_candidate = checkbox_candidates[0]
        
        logger.info(f"Edge detection found checkbox at ({best_candidate['x']}, {best_candidate['y']}) "
                   f"size: {best_candidate['w']}x{best_candidate['h']}")
        
        return (best_candidate['x'], best_candidate['y'], True)
        
    except Exception as e:
        logger.error(f"Error in edge detection: {e}")
        return (0, 0, False)


async def click_turnstile_by_offset(page: Page, max_attempts: int = 5) -> bool:
    """
    通过 iframe 位置 + 固定偏移量点击 Turnstile 复选框
    
    参数:
        page: Playwright Page 对象
        max_attempts: 最大尝试次数
    
    返回:
        bool: 是否成功
    """
    logger.info("Attempting to click Turnstile using iframe offset method...")
    
    # 等待 Turnstile 完全加载
    await asyncio.sleep(3)
    
    for attempt in range(max_attempts):
        try:
            # 查找 Turnstile iframe（使用更宽泛的选择器）
            iframe_locator = page.locator('iframe[src*="cloudflare"], iframe[title*="Widget"], iframe[id*="cf-"], iframe[name*="cf-"]')
            iframe_count = await iframe_locator.count()
            
            if iframe_count == 0:
                logger.info("✓ No Turnstile iframe found - verification may have passed")
                return True
            
            # 获取第一个匹配的 iframe 的位置和大小
            iframe_box = await iframe_locator.first.bounding_box(timeout=5000)
            
            if not iframe_box:
                logger.debug(f"Attempt {attempt + 1}: Could not get iframe bounding box")
                await asyncio.sleep(1)
                continue
            
            # Turnstile iframe 的标准尺寸约为 300x65 像素
            # 复选框通常在 iframe 左上角，偏移约 (30, 32) 像素
            # 这是 Cloudflare Turnstile 的标准布局
            checkbox_offset_x = 30  # 从 iframe 左边界的偏移
            checkbox_offset_y = 32  # 从 iframe 上边界的偏移
            
            # 计算复选框的绝对位置
            checkbox_x = iframe_box['x'] + checkbox_offset_x
            checkbox_y = iframe_box['y'] + checkbox_offset_y
            
            logger.info(f"Iframe at ({iframe_box['x']:.0f}, {iframe_box['y']:.0f}), size: {iframe_box['width']:.0f}x{iframe_box['height']:.0f}")
            logger.info(f"Calculated checkbox at ({checkbox_x:.0f}, {checkbox_y:.0f})")
            
            # 使用人类行为模拟移动鼠标
            await human_like_mouse_move(page, checkbox_x, checkbox_y, duration=random.uniform(0.5, 1.0))
            
            # 随机停顿
            await asyncio.sleep(random.uniform(0.2, 0.5))
            
            # 点击
            await page.mouse.click(checkbox_x, checkbox_y)
            logger.info("✓ Clicked Turnstile checkbox")
            
            # 等待验证完成
            await asyncio.sleep(4)
            
            # 检查是否成功（iframe 消失或有 token）
            iframe_count_after = await page.locator('iframe[src*="cloudflare"], iframe[title*="Widget"], iframe[id*="cf-"], iframe[name*="cf-"]').count()
            
            if iframe_count_after == 0:
                logger.info("✓ Verification completed! (iframe disappeared)")
                return True
            
            # 检查是否有验证令牌
            has_token = await page.evaluate("""
                () => {
                    const inputs = document.querySelectorAll('input[name^="cf-turnstile-response"]');
                    if (inputs.length > 0) {
                        const token = inputs[0].value;
                        return token && token.length > 0;
                    }
                    return false;
                }
            """)
            
            if has_token:
                logger.info("✓ Verification completed! (token found)")
                return True
            
            logger.debug(f"Attempt {attempt + 1}: Verification not completed yet")
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.debug(f"Attempt {attempt + 1}: {e}")
            await asyncio.sleep(1)
    
    logger.warning("Failed to click Turnstile after all attempts by offset")
    return False


async def click_turnstile_with_vision(page: Page, max_attempts: int = 5) -> bool:
    """
    使用 OpenCV 视觉识别并点击 Turnstile 复选框
    
    参数:
        page: Playwright Page 对象
        max_attempts: 最大尝试次数
    
    返回:
        bool: 是否成功
    """
    print("Attempting to click Turnstile using OpenCV vision...")
    
    # 创建临时目录
    temp_dir = Path("temp_screenshots")
    temp_dir.mkdir(exist_ok=True)
    
    # 等待更长时间让 Turnstile 完全加载
    print("Waiting for Turnstile to fully load...")
    # await asyncio.sleep(5)
    
    for attempt in range(max_attempts):
        try:
            # 截取整个页面
            screenshot_path = temp_dir / f"turnstile_opencv_{attempt}.png"
            with contextlib.suppress(Exception):
                await page.wait_for_selector("#cf-chl-widget-sdire_response", timeout=5000)
            await page.screenshot(path=str(screenshot_path))
            print(f"Screenshot saved: {screenshot_path}")
            
            # 使用 OpenCV 识别复选框位置（启用调试模式）
            x, y, found = find_checkbox_in_screenshot(str(screenshot_path), debug=True)
            
            if not found:
                print(f"Attempt {attempt + 1}: OpenCV could not find checkbox")
                
                # 如果是第一次尝试失败，再等待一下
                if attempt == 0:
                    print("First attempt failed, waiting longer for Turnstile...")
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(2)
                continue
            
            print(f"OpenCV found checkbox at ({x}, {y})")
            
            # 使用人类行为模拟移动鼠标
            await human_like_mouse_move(page, x, y, duration=random.uniform(0.5, 1.0))
            
            # 随机停顿
            await asyncio.sleep(random.uniform(0.2, 0.5))
            
            # 点击
            await page.mouse.click(x, y)
            print("✓ Clicked Turnstile checkbox")
            
            # 等待验证完成
            await asyncio.sleep(5)
            
            # 检查是否成功（检查验证令牌）
            has_token = await page.evaluate("""
                () => {
                    const inputs = document.querySelectorAll('input[name^="cf-turnstile-response"]');
                    if (inputs.length > 0) {
                        const token = inputs[0].value;
                        return token && token.length > 0;
                    }
                    return false;
                }
            """)
            
            if has_token:
                print("✓ Verification completed!")
                
                # 清理截图
                try:
                    screenshot_path.unlink()
                except:
                    pass
                
                return True
            else:
                print(f"Attempt {attempt + 1}: No token found after click")
            
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Attempt {attempt + 1}: {e}")
            await asyncio.sleep(2)
    
    print("Failed to click Turnstile after all attempts by opencv")
    return False


async def click_turnstile_human_like(page: Page, max_attempts: int = 20) -> bool:
    """
    使用计算机视觉识别并点击 Turnstile 复选框
    
    参数:
        page: Playwright Page 对象
        max_attempts: 最大尝试次数
    
    返回:
        bool: 是否成功
    """
    logger.info("Attempting to click Turnstile using computer vision...")
    
    # 创建临时目录
    temp_dir = Path("temp_screenshots")
    temp_dir.mkdir(exist_ok=True)
    
    for attempt in range(max_attempts):
        try:
            # 检查是否有 Turnstile iframe
            iframe_count = await page.locator('iframe[src*="challenges.cloudflare.com"]').count()
            
            if iframe_count == 0:
                logger.info("✓ No Turnstile iframe found - verification may have passed")
                return True
            
            # 截取整个页面
            screenshot_path = temp_dir / f"turnstile_{attempt}.png"
            await page.screenshot(path=str(screenshot_path))
            logger.debug(f"Screenshot saved: {screenshot_path}")
            
            # 使用 OpenCV 查找复选框
            x, y, found = find_checkbox_in_screenshot(str(screenshot_path))
            
            if not found:
                logger.debug(f"Attempt {attempt + 1}: Checkbox not found in screenshot")
                await asyncio.sleep(1)
                continue
            
            logger.info(f"Found checkbox at ({x}, {y})")
            
            # 使用人类行为模拟移动鼠标
            await human_like_mouse_move(page, x, y, duration=random.uniform(0.5, 1.0))
            
            # 随机停顿
            await asyncio.sleep(random.uniform(0.2, 0.5))
            
            # 点击
            await page.mouse.click(x, y)
            logger.info("✓ Clicked Turnstile checkbox")
            
            # 等待验证完成
            await asyncio.sleep(4)
            
            # 检查是否成功
            iframe_count_after = await page.locator('iframe[src*="challenges.cloudflare.com"]').count()
            if iframe_count_after == 0:
                logger.info("✓ Verification completed!")
                
                # 清理截图
                try:
                    screenshot_path.unlink()
                except:
                    pass
                
                return True
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.debug(f"Attempt {attempt + 1}: {e}")
            await asyncio.sleep(1)
    
    logger.warning("Failed to click Turnstile after all attempts by ocr")
    return False


def get_ip_location(ip: str) -> str:
    """获取 IP 地址的地理位置"""
    import requests
    try:
        # 使用 ip-api.com 免费 API
        response = requests.get(f"http://ip-api.com/json/{ip}?lang=zh-CN", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                country = data.get('country', 'Unknown')
                region = data.get('regionName', '')
                city = data.get('city', '')
                location = f"{country}"
                if region:
                    location += f" - {region}"
                if city:
                    location += f" - {city}"
                return location
    except Exception as e:
        logger.debug(f"Failed to get IP location: {e}")
    return "Unknown"

async def get_model_mapping(page: Page):
    src = """
    (function() {
        const modelNameMapping = {
            "Claude Haiku 4.5": "claude-haiku-4.5",
            "Claude Opus 4.5": "claude-opus-4.5",
            "Claude Opus 4.6": "claude-opus-4.6",
            "Claude Sonnet 4.5": "claude-sonnet-4.5",
            "Claude Sonnet 4.6": "claude-sonnet-4.6",
            "GPT 5": "gpt-5",
            "GPT 5.1": "gpt-5.1",
            "GPT 5.2": "gpt-5.2",
            "Grok 4 Fast": "grok-4-fast",
            "Grok 4 Fast Reasoning": "grok-4-fast-reasoning",
            "Grok 4.1 Fast": "grok-4.1-fast",
            "Grok 4.1 Fast Reasoning": "grok-4.1-fast-reasoning",
            "Grok Code Fast": "grok-code-fast",
            "Imagen 4": "imagen-4",
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Pro": "gemini-2.5-pro",
            "Gemini 3 Flash Preview": "gemini-3-flash-preview",
            "Gemini 3 Pro Preview": "gemini-3-pro-preview",
            "Gemini 3.1 Pro Preview": "gemini-3.1-pro-preview",
            "Mistral Large 3": "mistral-large-3",
            "Mistral Medium 3": "mistral-medium-3",
            "Mistral Medium 3.1": "mistral-medium-3.1",
        };
        
        const models = {};
        Array.from(document.querySelectorAll("[data-testid='playground-model-select-item']")).forEach(e => {
            const handlerId = e.dataset.value;
            const modelName = Array.from(e.querySelectorAll("[title]")).find(e => e.innerText).innerText;
            const mapped = modelNameMapping[modelName];
            if(mapped)
                models[mapped] = handlerId;
            else
                console.log("unknown models: ", modelName);
        });
        console.log(models);
        return models;
    })();
    """

    return await page.evaluate(src)

async def register():
    """执行自动化流程"""
    
    # 清除可能存在的环境变量代理，避免干扰
    import os
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
        if key in os.environ:
            del os.environ[key]
            logger.info(f"Cleared environment variable: {key}")
    
    proxy_config = None
    
    if USE_PROXY:
        logger.info("=" * 60)
        logger.info("Starting Browser with Proxy...")
        logger.info(f"Local proxy: {LOCAL_PROXY}")
        
        # 测试本地代理连接并获取 IP 信息
        logger.info("Testing proxy connectivity...")
        import requests
        proxy_ip = None
        location = "Unknown"
        
        try:
            # 使用本地代理测试
            proxies = {
                "http": LOCAL_PROXY,
                "https": LOCAL_PROXY
            }
            response = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=10)
            proxy_ip = response.json()['ip']
            
            # 获取 IP 地理位置
            location = get_ip_location(proxy_ip)
            
            logger.info(f"✓ Proxy test successful!")
            logger.info(f"  Proxy IP: {proxy_ip}")
            logger.info(f"  Location: {location}")
            logger.info("=" * 60)
            
            # 配置浏览器使用本地代理
            proxy_config = {
                "server": LOCAL_PROXY
            }
        except Exception as e:
            logger.error(f"✗ Proxy test failed: {e}")
            logger.error("Please check if v2ray is running on port 1080")
            logger.info("Continuing without proxy...")
            proxy_config = None
    else:
        logger.info("Starting Browser without Proxy...")
    
    async with AsyncCamoufox(
        headless=False,
        os="windows",
        geoip=False,  # 禁用自动 geoip 检测
        window=(1280, 720),
        humanize=False,
        locale="zh-CN",
        proxy=proxy_config,
    ) as browser:
        start_time = time.time()
        page: Page = await browser.new_page()

        # ========== 辅助函数 ==========
        
        async def dismiss_cookie_banner():
            """关闭 Cookie 同意弹窗"""
            try:
                for text in ['Accept', '接受', '同意', 'I agree', 'OK']:
                    btn = await page.query_selector(f"button:has-text('{text}')")
                    if btn and await btn.is_visible():
                        await btn.click()
                        print(f"Dismissed cookie banner: {text}")
                        await asyncio.sleep(0.5)
                        return
            except:
                pass
        
        async def click_btn(selectors: list, timeout=30000):
            """点击按钮，支持多选择器降级（CSS 和 XPath）"""
            await dismiss_cookie_banner()
            for selector in selectors:
                try:
                    # 处理 xpath 选择器
                    if selector.startswith('xpath='):
                        xpath = selector[6:]  # 去掉 'xpath=' 前缀
                        await page.locator(f"xpath={xpath}").click(timeout=timeout)
                    else:
                        await page.click(selector, timeout=timeout)
                    print(f"Clicked: {selector}")
                    return True
                except Exception as e:
                    continue
            print(f"Failed to click element {selectors}")
            return False
        
        async def fill_input(selectors: list, value: str):
            """填充输入框，支持多选择器降级（CSS 和 XPath）"""
            for selector in selectors:
                try:
                    if selector.startswith('xpath='):
                        xpath = selector[6:]
                        await page.locator(f"xpath={xpath}").fill(value)
                    else:
                        await page.fill(selector, value)
                    logger.info(f"Filled: {selector}")
                    return True
                except Exception as e:
                    continue
            print(f"Failed to fill: {value} to {selectors}")
            return False
        
        async def select_option(selectors: list, value: str):
            """选择下拉框选项，支持多选择器降级（CSS 和 XPath）"""
            for selector in selectors:
                try:
                    if selector.startswith('xpath='):
                        xpath = selector[6:]
                        await page.locator(f"xpath={xpath}").select_option(value)
                    else:
                        await page.select_option(selector, value)
                    logger.info(f"Selected: {value}")
                    return True
                except Exception as e:
                    continue
            print(f"Failed to select: {value} to {selectors}")
            return False
        
        async def wait_and_fill(selectors: list, value: str, timeout=30000):
            """等待元素出现后填充，支持多选择器降级（CSS 和 XPath）"""
            for selector in selectors:
                try:
                    if selector.startswith('xpath='):
                        xpath = selector[6:]
                        await page.locator(f"xpath={xpath}").wait_for(timeout=timeout)
                        await page.locator(f"xpath={xpath}").clear()
                        await page.locator(f"xpath={xpath}").fill(value)
                    else:
                        elem = await page.wait_for_selector(selector, timeout=timeout)
                        await page.locator(selector).clear()
                        await page.fill(selector, value)
                    logger.info(f"Filled: {selector}")
                    return True
                except:
                    continue
            print(f"Failed to wait and fill {value} to {selectors}")
            return False
        
        async def type_date(selectors: list, date_str: str, timeout=30000):
            """
            处理 React Aria DateField 等特殊日期选择器
            先点击激活，然后用键盘输入日期
            date_str 格式: YYYY-MM-DD 或 YYYYMMDD
            """
            # 清理日期格式，只保留数字
            date_digits = date_str.replace('-', '').replace('/', '')
            
            for selector in selectors:
                try:
                    if selector.startswith('xpath='):
                        xpath = selector[6:]
                        locator = page.locator(f"xpath={xpath}")
                    else:
                        locator = page.locator(selector)
                    
                    await locator.wait_for(timeout=timeout)
                    await locator.click()
                    await asyncio.sleep(0.3)
                    
                    # 全选并删除现有内容
                    await page.keyboard.press("Control+a")
                    await asyncio.sleep(0.1)
                    
                    # 逐个输入日期数字
                    for digit in date_digits:
                        await page.keyboard.type(digit, delay=50)
                    
                    logger.info(f"Typed date: {date_str}")
                    return True
                except Exception as e:
                    continue
            
            print(f"Failed to type date: {date_str} to {selectors}")
            return False

        # ========== 流程开始 ==========
        
        # 访问目标页面
        await page.goto("https://workspace.nexos.ai/authorization/login", timeout=60000)
        print("Page loaded")
        
        # 处理 Cookie 弹窗
        await dismiss_cookie_banner()
        
        # 步骤 1: 点击 "No account? Create o"
        SELECTOR_1 = [
            
          "[data-testid=\"login-page-sign-up-link\"]",
          "[data-discover=\"true\"]",
          "div.flex.flex-col:nth-child(1) > div.flex.flex-1:nth-child(2) > div.w-full.max-w-\\[380px\\] > a.inline-flex.items-center"

        ]
        await click_btn(SELECTOR_1, 1000)

        # 步骤 1.5: 先点击邮箱输入框，触发 Turnstile 加载
        SELECTOR_2 = [
            "[data-testid=\"auth-input-traits-email\"]",
            "#\\:r8\\:-form-item",
            "input[name=\"traits.email\"]"
        ]
        await click_btn(SELECTOR_2, 1000)
        
        # 等待 Turnstile 加载并尝试自动点击
        print("Waiting for Turnstile and attempting auto-click (page 1)...")
        await asyncio.sleep(3)
        
        # 直接使用 OCR 方法识别"确认"文字并点击
        verification_done = await click_turnstile_with_vision(page, max_attempts=3)
        
        if not verification_done:
            print("✓ No visible Turnstile or already verified")
            await asyncio.sleep(1)

        # 步骤 3: 输入邮箱
        SELECTOR_2 = [
            
          "[data-testid=\"auth-input-traits-email\"]",
          "#\\:r8\\:-form-item",
          "input[name=\"traits.email\"]"

        ]
        await click_btn(SELECTOR_2, 1000)

        # 步骤 3: 输入邮箱
        SELECTOR_3 = [
            
          "[data-testid=\"auth-input-traits-email\"]",
          "#\\:r8\\:-form-item",
          "input[name=\"traits.email\"]"

        ]
        # 使用 EmailService 邮箱服务（asyncio）
        email = await mail.create_temp_email_async()
        if not email:
            logger.error("邮箱申请失败，退出")
            return
        
        print(f"Created email: {email}")
        await wait_and_fill(SELECTOR_3, email)

        # 步骤 4: 点击 "Create account"
        SELECTOR_4 = [
            
          "[data-testid=\"auth-submit-method\"]",
          "button[name=\"method\"]",
          "div.flex.flex-col:nth-child(4) > div > form.flex.flex-col > button.inline-flex.items-center"

        ]
        await click_btn(SELECTOR_4, 1000)

        # 步骤 4.5: 等待 Turnstile 并尝试自动点击
        print("Waiting for Turnstile and attempting auto-click (page 2)...")
        await asyncio.sleep(3)
        
        # 直接使用 OCR 方法识别"确认"文字并点击
        verification_done = await click_turnstile_with_vision(page, max_attempts=3)
        
        if not verification_done:
            print("✓ No visible Turnstile or already verified")
            await asyncio.sleep(1)

        # 步骤 5: 点击 input
        SELECTOR_5 = [
            
          "[data-testid=\"auth-password-password\"]",
          "input[name=\"password\"]",
          "[data-rr-is-password=\"true\"]"

        ]
        await click_btn(SELECTOR_5, 1000)

        # 步骤 6: 输入密码
        SELECTOR_6 = [
          "[data-testid=\"auth-password-password\"]",
          "input[name=\"password\"]",
          "[data-rr-is-password=\"true\"]"
        ]
        # 生成随机密码 (8-12位，包含大小写字母、数字、特殊字符)
        password = generate_password(random.randint(8, 12))
        logger.info(f"Generated password: {password}")
        await wait_and_fill(SELECTOR_6, password)
        
        """
        # 步骤 6.5: 提交前再次检查 Turnstile
        logger.info("Checking Turnstile before submitting password...")
        await asyncio.sleep(2)
        
        # 直接使用 OCR 方法识别"确认"文字并点击
        verification_done = await click_turnstile_with_vision(page, max_attempts=3)
        
        if not verification_done:
            logger.info("✓ No visible Turnstile before submit")
            await asyncio.sleep(1)
        """

        # 步骤 7: 点击 "Continue"
        SELECTOR_7 = [
          "[data-testid=\"auth-submit-method\"]",
          "button[name=\"method\"]",
          "div.flex.flex-col:nth-child(3) > div:nth-child(1) > form.flex.flex-col > button.inline-flex.items-center"
        ]
        await click_btn(SELECTOR_7, 1000)
        
        # 步骤 8: 等待并获取邮箱确认链接
        # print("Waiting for confirmation email...")
        
        # 等待邮件到达（最多60秒）
        confirmation_link = None
        verification_code = None
        for attempt in range(24):  # 12次 x 5秒 = 60秒
            await asyncio.sleep(5)
            print(f"Attempt {attempt + 1}: Checking emails...")
            emails = await mail.get_emails_async()
            
            if emails:
                print(f"Found {len(emails)} email(s)")
                
                # 查找确认邮件
                for email_item in emails:
                    subject = email_item.get('subject', '')
                    html_content = quopri.decodestring(email_item.get('html_content', '')).decode('utf-8', errors='ignore')
                    text_content = quopri.decodestring(email_item.get('text_content', '')).decode('utf-8', errors='ignore')
                    
                    print(f"Email subject: {subject} ")
                    
                    # 从邮件内容中提取链接
                    import re
                    
                    links = re.findall(r'href="([^"]+)"', html_content, re.IGNORECASE)
                    for link in links:
                        if 'nexos.ai' in link and "https://" in link and len(link) > 100:
                            confirmation_link = link
                            print(f"Found confirmation link: {confirmation_link[:80]}... from html length: {len(confirmation_link)}")
                            break
                    if confirmation_link:
                        codes = re.findall(r'>\s*(\d{6})\s*<', text_content, re.IGNORECASE)
                        for code in codes:
                            if len(code) == 6:
                                verification_code = code
                                print(f"Found verification code: {verification_code} from text length: {len(verification_code)}")
                                break
                    
                    # 查找 nexos.ai 的链接
                    if not confirmation_link:
                        links = re.findall(r'(https://[\S]+)', text_content, re.IGNORECASE)
                        for link in links:
                            if 'nexos.ai' in link and len(link) > 100:
                                confirmation_link = link
                                print(f"Found confirmation link: {confirmation_link[:80]}... from text length: {len(confirmation_link)}")
                                break
                        if confirmation_link:
                            codes = re.findall(r'\s(\d{6})\s', text_content, re.IGNORECASE)
                            for code in codes:
                                if len(code) == 6:
                                    verification_code = code
                                    print(f"Found verification code: {verification_code} from text length: {len(verification_code)}")
                                    break
                    
                    
                if confirmation_link:
                    break
                else:
                    print("No confirmation link found in emails")
                    print(text_content)
                    print(html_content)
            else:
                print(f"Attempt {attempt + 1}: No emails yet")
        
        if not confirmation_link:
            print("Failed to get confirmation email after 60 seconds")
            print("Skipping email confirmation, will try to login directly")
        else:
            # 访问确认链接
            print("Clicking confirmation link...")
            await page.goto(confirmation_link, timeout=60000)
            await asyncio.sleep(3)
            
            if not verification_code:
                print("Failed to get verification code")
            else:
                print(f"Got verification code: {verification_code}")
                
                # 步骤 8.2: 输入验证码
                code_input_selector = [
                    "input[name='code']",
                    "[data-testid='verification-code-input']",
                    "input[type='text']"
                ]
                await wait_and_fill(code_input_selector, verification_code)
                await asyncio.sleep(1)
                
                # 步骤 8.3: 点击 Continue
                continue_selector = [".inline-flex", "button:has-text('Continue')"]
                await click_btn(continue_selector, 1000)
                await asyncio.sleep(3)
                
                logger.info("✓ Email confirmed!")
        
        # 步骤 9: 返回登录页面（无论是否成功确认邮箱）
        print("Going to login page...")
        
        # 如果当前不在登录页面，导航到登录页面
        if 'login' not in page.url:
            await page.goto("https://workspace.nexos.ai/authorization/login", timeout=60000)
            await asyncio.sleep(3)
        else:
            # 如果在验证成功页面，点击 Back to login
            try:
                back_selector = [".inline-flex", "button:has-text('Back to login')"]
                await click_btn(back_selector, timeout=5000)
                await asyncio.sleep(3)
            except:
                # 如果没有找到按钮，直接导航
                await page.goto("https://workspace.nexos.ai/authorization/login", timeout=60000)
                await asyncio.sleep(3)
        
        # 步骤 10: 先点击邮箱输入框，触发 Turnstile 加载
        logger.info("Logging in with registered account...")
        email_selector = [
            "#\\:r2\\:-form-item",
            "input[name='identifier']",
            "input[type='email']"
        ]
        await click_btn(email_selector)
        await asyncio.sleep(2)
        
        # 步骤 10.5: 处理登录页面的 Turnstile
        print("Handling Turnstile on login page...")
        verification_done = await click_turnstile_with_vision(page, max_attempts=3)
        
        if not verification_done:
            print("✓ No visible Turnstile on login page")
            await asyncio.sleep(1)
        
        # 步骤 11: 输入邮箱
        await wait_and_fill(email_selector, email)
        await asyncio.sleep(1)
        
        # 步骤 12: 输入密码
        password_selector = [
            ".pr-11",
            "input[name='password']",
            "input[type='password']"
        ]
        await wait_and_fill(password_selector, password)
        await asyncio.sleep(1)
        
        # 步骤 13: 点击登录按钮
        login_button_selector = [
            "button.inline-flex:nth-child(6)",
            "button[name='method']",
            "button:has-text('Sign in')"
        ]
        try:
            await click_btn(login_button_selector, 1000)
        except:
            ...
        await asyncio.sleep(5)
        
        print("✓ Login completed!")
        
        # 等待登录后的页面加载
        await asyncio.sleep(3)
        
        # 步骤 14: 获取 Chat ID
        print("Getting chat ID from URL...")
        current_url = page.url
        print(f"Current URL: {current_url}")
        
        chat_id = None
        # 从URL中提取chat ID（格式：https://workspace.nexos.ai/chat/{chat_id}）
        import re
        match = re.search(r'/chat/([a-f0-9\-]+)', current_url)
        if match:
            chat_id = match.group(1)
            print(f"Got chat ID: {chat_id}")
        else:
            print("Could not extract chat ID from URL")
            raise Exception("Could not extract chat ID from URL")
        
        # 步骤 15: 获取并保存 Cookie
        print("Getting cookies...")
        cookies = await page.context.cookies()
        
        # 转换为字符串格式：name=value; name2=value2
        cookie_string = '; '.join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
        
        # 转换为字典格式
        cookie_dict = {}
        for cookie in cookies:
            cookie_dict[cookie['name']] = cookie['value']
        
        print(f"Got {len(cookies)} cookies")
        
        print("Clicking accept TOS...")
        accept_tos_selector = [
            "xpath=//*[text()='Continue']"
        ]
        try:
            await click_btn(accept_tos_selector, 10000)
        except:
            ...
        
        print("Clicking skip tutorial...")
        skip_tutorial_selector = [
            "xpath=//*[text()='Skip & Close']"
        ]
        try:
            await click_btn(skip_tutorial_selector, 15000)
        except:
            ...
        
        print("Clicking welcome close...")
        welcome_selector = [
            "button.top-4.right-4"
        ]
        try:
            await click_btn(welcome_selector, 10000)
        except:
            ...
        
        print("Clicking model choose...")
        model_choose_selector = [
            "[data-testid=chat-prompt-model-selector-trigger]"
        ]
        try:
            await click_btn(model_choose_selector, 10000)
        except:
            ...
        
        print("Clicking model list...")
        model_list_selector = [
            "xpath=//*[text()='All models and agents']"
        ]
        try:
            await click_btn(model_list_selector, 10000)
        except:
            ...

        model_mapping = await get_model_mapping(page)

        # 保存账号信息（包含cookie字符串、字典和chat ID）
        account_data = {
            "email": email,
            "password": password,
            "chat_id": chat_id,  # Chat ID
            "cookies": cookie_string,  # Cookie字符串格式（用于HTTP请求头）
            "cookie_dict": cookie_dict,  # Cookie字典格式（方便程序使用）
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_mapping": model_mapping,
        }
        
        # 保存到文件
        import json
        from pathlib import Path
        
        output_file = Path("demo/nexos_accounts.json")
        # output_file.mkdir(parents=True, exist_ok=True)
        
        # 读取现有账号（如果文件存在）
        accounts = []
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    accounts = json.load(f)
            except:
                accounts = []
        
        # 添加新账号
        accounts.append(account_data)
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(accounts, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Account saved to {output_file}")
        print(f"  Email: {email}")
        print(f"  Password: {password}")
        print(f"  Chat ID: {chat_id if chat_id else 'N/A'}")
        print(f"  Cookies: {len(cookie_string)} characters")

        # ========== 流程结束 ==========
        
        logger.info("=" * 60)
        logger.info("Registration and Login completed successfully!")
        logger.info("=" * 60)
        if 'email' in dir():
            logger.info(f"Email: {email}")
        if 'password' in dir():
            logger.info(f"Password: {password}")
        if 'chat_id' in dir() and chat_id:
            logger.info(f"Chat ID: {chat_id}")
        if 'cookie_string' in dir():
            logger.info(f"Cookies: {len(cookie_string)} characters")
        logger.info(f"Account data saved to: demo/nexos_accounts.json")
        logger.info("=" * 60)
        
        end_time = time.time()
        logger.info(f"Total time: {end_time - start_time:.2f} seconds")
        
        # 等待用户按回车键关闭浏览器（必须在 with 块内）
        # input("\n按回车键关闭浏览器...")

async def run():
    for i in range(10):
        print(f"Registering account {i + 1}...")
        try:
            await register()
        except Exception as e:
            print(f"Error: {e} in {e.__traceback__}")

if __name__ == "__main__":
    asyncio.run(run())
