#include "ros/ros.h"

struct PID
{
    float last_error;
    float new_error;
    float integral;
    float pout;
    float iout;
    float dout;
};


class PIDcontroler
{
public:
    PIDcontroler(float maxOutput = 0.4, float integralLimit = 0.3, float deadband = 0.02, float kp = 0.7, float ki = 0.0, float kd = 0.0)
    : _kp(kp),
    _ki(ki),
    _kd(kd),
    _maxOutput(abs(maxOutput)),
    _integralLimit(abs(integralLimit)),
    _deadband(abs(deadband))
    {
        _pid.integral = 0;
        _pid.last_error = 0;
        _pid.new_error = 0;
        _pid.pout = 0;
        _pid.iout = 0;
        _pid.dout = 0;
    }
    void PIDreset(float maxOutput = 0.7, float integralLimit = 0.3, float deadband = 0.02, float kp = 0.7, float ki = 0.0, float kd = 0.0)
    {
        _kp = kp;
        _ki = ki;
        _kd = kd;
        _maxOutput = abs(maxOutput);
        _integralLimit = abs(integralLimit);
        _deadband = abs(deadband);
    }

    // ===================== 【新增】适配你的PID重置函数（核心） =====================
    void reset() {
        _pid.integral = 0;      // 清空积分项（解决PID饱和）
        _pid.last_error = 0;    // 重置上一帧误差
        _pid.new_error = 0;     // 重置当前误差
        _pid.pout = 0;
        _pid.iout = 0;
        _pid.dout = 0;
    }

    float PIDcalculate(float target, float measure, double dt)
    {
        if(dt <= 0)
        {
            return 0;
        }
        float dvalue = target - measure;
        // std::cout<< "dvalue"<< dvalue << std::endl;
        if(dvalue < _deadband && dvalue > -_deadband)
        {
            dvalue = 0;
        }
        _pid.new_error = dvalue;
        _pid.pout = _kp*dvalue;
        _pid.integral += _ki*(dvalue)*dt;
        _pid.dout = _kd*(_pid.new_error - _pid.last_error) / dt;
        if(_pid.integral > _integralLimit)
        {
            _pid.integral = _integralLimit;
        }
        else if(_pid.integral < -_integralLimit)
        {
            _pid.integral = -_integralLimit;
        }
        float output = _pid.pout + _pid.integral + _pid.dout;
        if(output > _maxOutput)
        {
            return _maxOutput;
            // std::cout<< "_maxOutput" << _maxOutput << std::endl;
        }
        else if(output < -_maxOutput)
        {
            return -_maxOutput;
            // std::cout<< "-_maxOutput" << -_maxOutput << std::endl;
        }
        _pid.last_error = dvalue;
        // std::cout<< "output" << output << std::endl;
        return output;
    }
private:
PID _pid;
float _maxOutput;
float _integralLimit;
float _IntegralSeparate = 0;
float _deadband = 0.01;
float _max_err = 10;
float _kp;
float _ki;
float _kd;
};