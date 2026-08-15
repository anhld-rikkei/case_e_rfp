## Case 1: case_a_manuals

Case này nặng về xử lý dữ liệu và thuần về RAG nhất

Case này thực hiện đọc các file pdf trong thư mục case_a_manuals/mitsubishi -> md
Xây dựng chatbot hỏi đáp trên tài liệu

Đầu vào: Câu hỏi của người dùng
Đầu ra: Câu trả lời của chatbot + Citation đoạn chunk được sử dụng

Gợi ý: sử dụng các kỹ thuật RAG khác nhau ở tuần 3 (Buổi 7 - 9) kết hợp với các kỹ thuật RAG nâng cao (đồ thị tri thức, có thể sử dụng các thư viện GraphRAG, LightRAG hoặc tự phát triển đồ thị tri thức riêng)

## Case 2: case_c_tickets

Case này liên quan đến trả lời ticket dịch vụ trên công ty cùng với đó là phân loại ticket, đánh giá mức độ ưu tiên và đánh giá khả năng AI tự giải quyết ticket (có thể tự giải quyết hoặc cần chuyển cho con người đối với các ticket khó liên quan đển bảo mật chẳng hạn)

Data:
case_c_tickets/synthetic/kb/kb_articles.jsonl: Nguồn tri thức để hướng dẫn AI xử lý
case_c_tickets/synthetic/tickets.jsonl: Các ticket trong quá khứ, giúp xác định các loại ticket đẻ phân loại, đánh giá mức độ ưu tiên

Đầu vào: Yêu cầu của người dùng
Đầu ra: Câu trả lời của AI đề xuất giải pháp + phân loại ticket (category) + flag đánh dấu có cần con người vào confirm giải pháp hay không.

## Case 3: case_d_callcenter

Case này liên quan đến việc đưa ra đưa ra gợi ý cho nhân viên để trả lời khách hàng.

Data:
case_d_callcenter/synthetic/sakura_mart_kb.jsonl: Nguồn tri thức để gợi ý cho nhân viên.
case_d_callcenter/synthetic/bsd_phone_calls_asr.jsonl, case_d_callcenter/bsd_phone_calls.jsonl: các data này dùng để lọc, làm sạch và để test kết quả của hệ thống.

Đầu vào: Câu hỏi của người dùng
Đầu ra: câu gợi ý trả lời cho nhân viên.

Gợi ý: Case này tương đối đơn giản về luồng RAG nhưng cần chú trọng về mặt thời gian đưa ra gợi ý (thời gian đưa ra gợi ý nên nằm trong khoảng 1-2s). Khuyến khích demo thêm phần Text to speech để hình dung rõ hơn về mặt thời gian và tăng độ chân thực(recommend pipeline: retrieve tri thức -> tạo ra gợi ý trả lời -> sinh ra câu trả lời dạng streaming response -> sử dụng các dịch vụ TTS chuyển văn bản thành giọng nói (tìm hiểu thêm không bắt buộc))

## Case 4: case_e_rfp

Case này liên quan đến việc gen ra 1 văn bản (hồ sơ thầu) không phải chatbot hỏi đáp

Data:
case_e_rfp/synthetic/proposals: Các hồ sơ thầu trong quá khứ đã thực hiện, dùng làm nguồn tri thức
case_e_rfp/synthetic/certs/fake_iso27017.txt: Chứng chỉ fake của công ty, dùng để đảm bảo hồ sơ thầu được gen ra không được chứa chứng chỉ này
case_e_rfp/synthetic/rfps: Các requirement của dự án mà công ty tham gia
v2/Data for use case/case_e_rfp/synthetic/capability_sheet.json: Năng lực thực tế của công ty.

Đầu vào: các bản requirement của dự án
Đầu ra: Hồ sơ đầu thầu đảm bảo đáp ứng được các requirement của dự án, phù hợp với năng lực công ty



Lưu ý chung: Trong quá trình dev khuyến khích sử dụng các phương pháp khác nhau về RAG và agent, sử dụng các framework RAGAS hay deepeval để so sánh kết quả để đưa ra lý do vì sao dùng phương án.
Khuyến khích demo giao diện nếu có
Chúc mọi người làm project thành công ạ!