import docker
import subprocess
import os
import utils.logger as logger



docker_client = None



def _initialize_docker():
    logger.info("Initializing Docker...")
    global docker_client
    
    # 尝试多种方式连接 Docker
    connection_methods = [
        # 方法 1: 使用默认环境配置（DOCKER_HOST 环境变量或默认 socket）
        lambda: docker.from_env(),
        # 方法 2: 尝试 Docker Desktop 的 socket 路径（macOS）
        lambda: docker.DockerClient(base_url='unix://' + os.path.expanduser('~/.docker/run/docker.sock')),
        # 方法 3: 尝试传统的 Unix socket 路径
        lambda: docker.DockerClient(base_url='unix:///var/run/docker.sock'),
    ]
    
    last_error = None
    for i, method in enumerate(connection_methods, 1):
        try:
            logger.info(f"尝试连接方式 {i}...")
            docker_client = method()
            # 测试连接是否有效
            docker_client.version()
            logger.info("Docker 初始化成功")
            return
        except Exception as e:
            logger.warning(f"连接方式 {i} 失败: {e}")
            last_error = e
            continue
    
    # 所有方法都失败
    error_msg = (
        f"无法连接到 Docker daemon。\n"
        f"最后尝试的错误: {last_error}\n"
        f"请确保：\n"
        f"  1. Docker Desktop 已启动并正在运行\n"
        f"  2. Docker daemon 正在运行（尝试运行 'docker ps' 测试）\n"
        f"  3. 您有访问 Docker socket 的权限"
    )
    logger.fatal(error_msg)



def get_docker_client():
    """
    Gets the Docker client. If it is not initialized, initializes it (once per program).
    """

    if docker_client is None:
        _initialize_docker()
    
    return docker_client



def build_docker_image(dockerfile_dir: str, tag: str) -> None:
    """
    Builds a Docker image.

    Args:
        dockerfile_dir: Path to a directory containing a Dockerfile
        tag: Tag to give the Docker image
    """

    logger.info(f"Building Docker image: {tag}")
    
    try:
        # Capture output to help with debugging
        result = subprocess.run(
            ["docker", "build", "--network=host", "-t", tag, dockerfile_dir],
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully built Docker image: {tag}")
        else:
            raise Exception(f"Docker build failed with exit code {result.returncode}")
            
    except Exception as e:
        logger.error(f"Failed to build Docker image: {e}")
        raise



def get_num_docker_containers() -> int:
    """
    Gets the number of Docker containers running.
    """

    # This is equivalent to `docker ps -q | wc -l`
    result = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=1)
    return len([line for line in result.stdout.strip().split('\n') if line.strip()])



# TODO ADAM: optimize
def stop_and_delete_all_docker_containers() -> None:
    """
    Stops and deletes all Docker containers.
    """

    docker_client = get_docker_client()
    
    logger.info("Stopping and deleting all containers...")
    
    for container in docker_client.containers.list(all=True):
        logger.info(f"Stopping and deleting container {container.name}...")

        try:
            container.stop(timeout=3)
        except Exception as e:
            logger.warning(f"Could not stop container {container.name}: {e}")
        
        try:
            container.remove(force=True)
        except Exception as e:
            logger.warning(f"Could not remove container {container.name}: {e}")

        logger.info(f"Stopped and deleted container {container.name}")

    docker_client.containers.prune()
    
    logger.info("Stopped and deleted all containers")



def create_internal_docker_network(name: str) -> None:
    """
    Creates an internal Docker network, if it does not already exist.
    """

    docker_client = get_docker_client()
    
    try:
        docker_client.networks.get(name)
        logger.info(f"Found internal Docker network: {name}")
    except docker.errors.NotFound:
        docker_client.networks.create(name, driver="bridge", internal=True)
        logger.info(f"Created internal Docker network: {name}")



def connect_docker_container_to_internet(container: docker.models.containers.Container) -> None:
    """
    Connects a Docker container to the internet.
    """

    docker_client = get_docker_client()

    logger.info(f"Connecting Docker container {container.name} to internet...")

    bridge_network = docker_client.networks.get("bridge")
    bridge_network.connect(container)
    
    logger.info(f"Connected Docker container {container.name} to internet")