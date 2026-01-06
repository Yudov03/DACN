"""
Script để tạo synthetic dataset cho testing
Tạo variations của files hiện có để mở rộng dataset
"""

import sys
import os
import io
import shutil
from pathlib import Path
import random

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def create_synthetic_pdfs(source_pdf: Path, output_dir: Path, count: int = 10):
    """
    Tạo copies của PDF với metadata khác nhau
    (Trong thực tế sẽ modify content, nhưng để test ta chỉ copy)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = source_pdf.stem
    ext = source_pdf.suffix

    variations = [
        "Quy định học phí năm 2023",
        "Quy định học phí năm 2024",
        "Quy định học phí năm 2025",
        "Hướng dẫn đăng ký học phần HK1-2024",
        "Hướng dẫn đăng ký học phần HK2-2024",
        "Thông báo lịch thi giữa kỳ",
        "Thông báo lịch thi cuối kỳ",
        "Quy chế đào tạo đại học 2023",
        "Quy chế đào tạo đại học 2024",
        "Hướng dẫn viết báo cáo thực tập",
    ]

    created_files = []
    for i in range(min(count, len(variations))):
        new_name = f"{variations[i]}{ext}"
        dest = output_dir / new_name
        shutil.copy2(source_pdf, dest)
        created_files.append(dest)
        print(f"  Created: {new_name}")

    return created_files

def create_synthetic_text_files(output_dir: Path, count: int = 15):
    """
    Tạo các file text với nội dung synthetic về giáo dục
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = [
        {
            "name": "Quy định điểm danh HK1-2024.txt",
            "content": """QUY ĐỊNH VỀ ĐIỂM DANH SINH VIÊN
Học kỳ 1 năm học 2024-2025

1. Quy định chung
- Sinh viên phải điểm danh đầu giờ mỗi buổi học
- Được phép vắng tối đa 20% số buổi học
- Vắng quá 20% sẽ bị cấm thi

2. Hình thức điểm danh
- Điểm danh bằng thẻ sinh viên
- Giảng viên ghi nhận trong hệ thống
- Sinh viên kiểm tra trên portal

3. Xin phép vắng
- Nộp đơn trước 1 ngày
- Có lý do chính đáng
- Được duyệt qua email

Có hiệu lực từ 01/09/2024
"""
        },
        {
            "name": "Học phí HK1-2024.txt",
            "content": """THÔNG BÁO HỌC PHÍ HỌC KỲ 1 NĂM HỌC 2024-2025

Mức học phí:
- Hệ đại trà: 15.000.000 VNĐ/học kỳ
- Hệ chất lượng cao: 25.000.000 VNĐ/học kỳ
- Hệ liên thông: 12.000.000 VNĐ/học kỳ

Thời hạn đóng:
- Đợt 1: 01/09 - 15/09/2024
- Đợt 2: 16/09 - 30/09/2024
- Phí trễ hạn: 100.000 VNĐ

Hình thức đóng:
- Chuyển khoản qua ngân hàng
- Đóng trực tiếp tại phòng Tài vụ
- Qua cổng thanh toán điện tử

Liên hệ: phong.taive@university.edu.vn
"""
        },
        {
            "name": "Lịch thi giữa kỳ HK1-2024.txt",
            "content": """LỊCH THI GIỮA KỲ HỌC KỲ 1 NĂM 2024

Thời gian: Từ 15/10 đến 25/10/2024

Các môn thi:
1. Toán cao cấp: 15/10, 8h-10h, Phòng A101
2. Vật lý đại cương: 16/10, 14h-16h, Phòng B205
3. Lập trình Python: 18/10, 8h-10h, Phòng C301
4. Cơ sở dữ liệu: 20/10, 14h-16h, Phòng D102
5. Mạng máy tính: 22/10, 8h-10h, Phòng E201

Quy định:
- Sinh viên mang thẻ SV và CMND
- Có mặt trước 15 phút
- Không mang tài liệu
- Điện thoại tắt nguồn

Tra cứu: portal.university.edu.vn
"""
        },
        {
            "name": "Hướng dẫn đăng ký môn học.txt",
            "content": """HƯỚNG DẪN ĐĂNG KÝ MÔN HỌC

Bước 1: Đăng nhập portal
- Truy cập portal.university.edu.vn
- Đăng nhập bằng mã SV và mật khẩu
- Chọn "Đăng ký môn học"

Bước 2: Chọn môn học
- Xem danh sách môn mở
- Kiểm tra điều kiện tiên quyết
- Chọn lớp phù hợp với lịch

Bước 3: Xác nhận
- Kiểm tra lại môn đã chọn
- Bấm "Xác nhận đăng ký"
- In phiếu đăng ký

Lưu ý:
- Đăng ký đúng thời gian quy định
- Tối thiểu 12 tín chỉ/kỳ
- Tối đa 24 tín chỉ/kỳ
"""
        },
        {
            "name": "Quy định làm bài tập lớn.txt",
            "content": """QUY ĐỊNH VỀ BÀI TẬP LỚN

1. Yêu cầu chung
- Làm theo nhóm 3-5 sinh viên
- Nộp đúng deadline
- Trình bày báo cáo + demo

2. Hình thức nộp bài
- File báo cáo PDF
- Source code (nếu có)
- Slide thuyết trình
- Upload lên hệ thống LMS

3. Tiêu chí đánh giá
- Nội dung: 50%
- Trình bày: 20%
- Demo: 20%
- Câu hỏi: 10%

4. Thời gian
- Giao đề: Tuần 3
- Deadline: Tuần 12
- Báo cáo: Tuần 13-14

Liên hệ GVHD nếu có thắc mắc
"""
        },
    ]

    created_files = []
    for i, template in enumerate(templates):
        if i >= count:
            break
        file_path = output_dir / template["name"]
        file_path.write_text(template["content"], encoding='utf-8')
        created_files.append(file_path)
        print(f"  Created: {template['name']}")

    # Tạo thêm variations
    for i in range(len(templates), count):
        name = f"Document_{i+1:03d}.txt"
        content = f"""TÀI LIỆU {i+1}

Đây là tài liệu test số {i+1} được tạo tự động.
Nội dung này được dùng để test hệ thống retrieval.

Thông tin chi tiết:
- Document ID: DOC{i+1:03d}
- Ngày tạo: 2024-12-28
- Loại: Text document
- Mục đích: Testing

Nội dung bao gồm các thông tin về:
1. Quy định học vụ
2. Hướng dẫn sinh viên
3. Thông báo từ nhà trường
4. Tài liệu tham khảo

Liên hệ: info@university.edu.vn
"""
        file_path = output_dir / name
        file_path.write_text(content, encoding='utf-8')
        created_files.append(file_path)

    return created_files

def main():
    print("=" * 70)
    print("CREATING SYNTHETIC DATASET")
    print("=" * 70)

    resource_dir = PROJECT_ROOT / "data" / "resource"

    # 1. Tạo synthetic PDFs
    print("\n[1] Creating synthetic PDF files...")
    pdf_source = resource_dir / "documents" / "QD 2349_ban hành Quy định Quản lý cấp phát văn bằng chứng chỉ năm 2024.pdf"

    if pdf_source.exists():
        pdf_files = create_synthetic_pdfs(
            pdf_source,
            resource_dir / "documents",
            count=10
        )
        print(f"  ✓ Created {len(pdf_files)} PDF files")
    else:
        print(f"  ✗ Source PDF not found: {pdf_source}")

    # 2. Tạo synthetic text files
    print("\n[2] Creating synthetic text files...")
    text_files = create_synthetic_text_files(
        resource_dir / "documents",
        count=15
    )
    print(f"  ✓ Created {len(text_files)} text files")

    # 3. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_new = len(pdf_files) + len(text_files)
    print(f"Total new files created: {total_new}")
    print(f"  - PDFs: {len(pdf_files)}")
    print(f"  - Text: {len(text_files)}")
    print(f"\nFiles saved to: {resource_dir / 'documents'}")

    print("\n[NEXT STEP] Run import script:")
    print("  python scripts/import_resources.py --clear")
    print("=" * 70)

if __name__ == "__main__":
    main()
