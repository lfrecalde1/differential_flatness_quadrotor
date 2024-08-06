from setuptools import find_packages, setup

package_name = 'differential_flatness'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fer',
    maintainer_email='fernandorecalde@uti.edu.ec',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "planning = differential_flatness.main:main",
            "flatness_controller = differential_flatness.main_controller:main",
            "hopf_controller = differential_flatness.main_hopf_vibration:main"
        ],
    },
)
