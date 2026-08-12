"""Benchmark tool for measuring JD & Search Query processing latency and vector similarity."""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

from ai_core.embeddings.bge_m3 import BgeM3Embedder
from ai_core.schemas import CVProfile, Experience, Skill
from ai_core.search import analyze_query_or_jd


def _cosine(u: list[float], v: list[float]) -> float:
    return sum(a * b for a, b in zip(u, v)) / (
        math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v))
    )


async def run_benchmark() -> None:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("==========================================================================")

    print("             SMART-CV-AI-CORE : JD & QUERY PIPELINE BENCHMARK             ")
    print("==========================================================================")

    embedder = BgeM3Embedder()

    # Sample inputs
    short_query = "Quản trị hệ thống DevOps 2 năm kinh nghiệm K8s AWS Docker CI/CD ELK Grafana Prometheus Shell Script"
    long_jd = (
        "Mô tả công việc\n"
        "Triển khai cài đặt, cấu hình, tối ưu các hệ thống phần mềm như hệ điều hành, csdl, web server, app server, message queue, các hệ thống cân bằng tải, các hệ thống quản lý logs, các hệ thống monitoring, các hệ thống bảo mật, Data Platform, AI Platform, nền tảng cho devops, ci/cd, container và các hệ thống giải pháp khác khi có yêu cầu.\n"
        "Triển khai hạ tầng hybrid giữa on-prem và public cloud.\n"
        "Làm việc với khách hàng, bộ phận phân tích nghiệp vụ để chuyển các yêu cầu sang thiết kế kỹ thuật. \n"
        "Triển khai các giải pháp tự động hóa trong doanh nghiệp như Workload Automation, APM, AI OPs\n"
        "Triển khai các giải pháp tích hợp trong doanh nghiệp như APIs, ESB.\n"
        "Triển khai các nền tảng dữ liệu lớn như datalake, lakehouse, hạ tầng cho AI.\n"
        "Viết các shell script để tự động hóa hoặc đáp ứng yêu cầu quản trị, vận hành các hệ thống CNTT. \n"
        "Yêu cầu ứng viên\n"
        "Tốt nghiệp ĐH, chuyên ngành CNTT\n"
        "Có ít nhất 2 năm kinh nghiệm liên quan đến công việc quản trị vận hành hệ thống. \n"
        "Có kinh nghiệm triển khai hạ tầng các hệ thống như Monitoring, Logs\n"
        "Ưu tiên\n"
        "Có kiến thức về bảo mật, sql. Hiểu rõ SSO/SAML 2.0, LDAP, Active Directory, DNS \n"
        "Có kinh nghiệm triển khai các hệ thống CNTT trên Cloud như AWS, Azure, GCP, OCI, IBM\n"
        "Có kinh nghiệm viết shell script để thực hiện tự động hóa các tác vụ quản trị hệ thống.\n"
        "Có kinh nghiệm với các hệ thống Containers như K8s, Openshift.\n"
        "Có kinh nghiệp sử dụng các công cụ DevOps, ELK stack, Grafana, Prometheus, Git, CI/CD, Jenkins.\n"
        "Có khả năng lựa chọn giải pháp và công nghệ phù hợp dựa trên budgets, kiến trúc hiện tại và nhu cầu kinh doanh.\n"
        "Có khả năng thích nghi nhanh với thay đổi của công nghệ, các môi trường kiến trúc phức tạp\n"
        "Trách nhiệm, ham học hỏi, kỹ năng phân tích và giải quyết vấn đề tốt\n"
        "Tiếng Anh: Toeic min 500\n"
        "Quyền lợi\n"
        "Lương: Thỏa thuận + các khoản thưởng cuối năm, thưởng HQKD (14-16 tháng lương/năm)\n"
        "Lộ trình phát triển và thăng tiến rõ ràng với kế hoạch đào tạo bài bản.\n"
        "Làm việc với đồng đội chuyên gia, xuất sắc trong lĩnh vực System, Cloud.\n"
        "Ngân sách đào tạo lên đến $5000/năm, bao gồm chi phí tài trợ học & thi các chứng chỉ quốc tế.\n"
        "Hưởng mức lương cạnh tranh, thưởng hấp dẫn trực tiếp từ lợi nhuận Công ty (cam kết tối thiểu 14 tháng lương/năm).\n"
        "Gói Bảo hiểm sức khỏe cho bản thân và gia đình lên đến $3000/người.\n"
        "Làm việc trong môi trường chuyên nghiệp, thân thiện với 4 giá trị cốt lõi: Trust, Teamwork, Knowledge & Creativity, Customer.\n"
        "Sống & làm việc cân bằng với 4 giá trị: Health, Heart, Mind, Spirit cùng chính sách làm việc linh hoạt.\n"
        "Mối quan hệ đồng hành giữa sếp & nhân viên. Công ty là ngôi nhà thứ hai nơi các thành viên sống và làm việc cùng nhau với trái tim nhiệt huyết, chân thành và tinh thần học hỏi không ngừng\n"
        "Các hoạt động văn hóa, giải trí phong phú: Team building, CLB Thể thao, âm nhạc, chương trình Sun-flower, Happy Hour..."
    )

    # Synthetic System / DevOps Engineer CV Profile matching the JD
    matching_profile = CVProfile(
        candidate_name="Nguyễn Văn A",
        headline="System Engineer / DevOps Engineer",
        skills=[
            Skill(name="DevOps", canonical_name="DevOps", confidence=1.0),
            Skill(name="Docker", canonical_name="Docker", confidence=1.0),
            Skill(name="Kubernetes", canonical_name="Kubernetes", confidence=1.0),
            Skill(name="AWS", canonical_name="AWS", confidence=1.0),
            Skill(name="CI/CD", canonical_name="CI/CD", confidence=1.0),
            Skill(name="Prometheus", canonical_name="Prometheus", confidence=1.0),
            Skill(name="Grafana", canonical_name="Grafana", confidence=1.0),
            Skill(name="Linux", canonical_name="Linux", confidence=1.0),
        ],
        experiences=[
            Experience(
                job_title="DevOps Engineer",
                company="System & Cloud Solutions",
                description=[
                    "Triển khai hạ tầng Cloud AWS, Kubernetes (K8s), Docker Compose và CI/CD với Jenkins.",
                    "Xây dựng hệ thống Monitoring & Logging với Prometheus, Grafana, ELK stack.",
                    "Viết Shell script tự động hóa quản trị vận hành hệ thống Linux."
                ],
                skills=["DevOps", "Docker", "Kubernetes", "AWS", "Prometheus", "Grafana", "Linux"],
                is_current=True,
            )
        ],
    )

    # 1. Warmup Embedder
    _ = embedder.embed_text("warmup query")

    # 2. Benchmark Short Query
    start = time.perf_counter()
    res_short = await analyze_query_or_jd(short_query, embedder)
    dur_short = (time.perf_counter() - start) * 1000

    # 3. Benchmark Long JD
    start = time.perf_counter()
    res_long = await analyze_query_or_jd(long_jd, embedder)
    dur_long = (time.perf_counter() - start) * 1000

    # 4. Generate Candidate CV Profile Embedding Vector
    cv_embed_res = embedder.embed_profile(matching_profile)

    # 5. Calculate Cosine Similarity
    sim_short_cv = _cosine(res_short.embedding.vector, cv_embed_res.vector)
    sim_long_cv = _cosine(res_long.embedding.vector, cv_embed_res.vector)

    print("\n--- LATENCY METRICS ---")
    print(f"  Short Query Latency:  {dur_short:.2f} ms")
    print(f"  Long JD Latency:       {dur_long:.2f} ms")

    print("\n--- EXTRACTED FILTERS ---")
    print(f"  Short Query Skills:    {res_short.filters.skills}")
    print(f"  Short Query Titles:    {res_short.filters.job_titles}")
    print(f"  Long JD Skills:        {res_long.filters.skills}")
    print(f"  Long JD Titles:        {res_long.filters.job_titles}")

    print("\n--- VECTOR SPACE SIMILARITY (Cosine Similarity on pgvector) ---")
    print(f"  Cosine Sim (Short Query <-> Candidate CV): {sim_short_cv * 100:.2f}%")
    print(f"  Cosine Sim (Long JD       <-> Candidate CV): {sim_long_cv * 100:.2f}%")
    print("==========================================================================")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
