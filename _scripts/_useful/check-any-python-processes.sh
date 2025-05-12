while true; do
  sleep 60
  python_process_count=$(ps -aux | grep python | grep -v "grep" | wc -l)
  echo "Number of Python processes running: $python_process_count"
  if [ "$python_process_count" -eq 0 ]; then
    echo "No Python processes running. Stopping script."
    break
  fi
done
