# 파일명: push_all.sh
rm -f .git/index.lock  # 잠금파일 삭제
git add .
git commit -m "업데이트: coco_text_pairs 전체 포함"
git push origin main