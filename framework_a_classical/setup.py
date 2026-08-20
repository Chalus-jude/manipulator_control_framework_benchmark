from setuptools import find_packages, setup

package_name = 'framework_a_classical'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='Ch',
    maintainer_email='you@example.com',
    description='Framework A: classical Jacobian IK + velocity-based PID control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller_node = framework_a_classical.controller_node:main',
            'test_client = framework_a_classical.test_client:main',
        ],
    },
)